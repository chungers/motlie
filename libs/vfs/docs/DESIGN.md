# motlie-vfs: Transport-Agnostic Virtual Filesystem Library

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-28 | @claude | Reframe overlay as file-driven memfs layer: whiteout/tombstone semantics, transparent cross-boundary rename, OverlayEntryKind enum |
| 2026-03-27 | @claude | Address PR #115 review: synthetic directories, policy-filtered readdir, mutation semantics table |
| 2026-03-27 | @claude | Initial DESIGN: composable architecture (core + vsock + rpc), file-level in-memory overlay with layered injection, pluggable control frontends (Unix socket / HTTP / in-process), VMM guest integration with SSH overlay example, alternatives analysis |

## Problem Statement

motlie-vmm's architecture routes all guest filesystem I/O through a host-side vsock FS server
(see `motlie-vmm` design doc, sections 10-12). The current design hardwires the FUSE client to
vsock inside a VM and the FS server to the VMM daemon process. This coupling prevents three
valuable deployment models:

1. **Linux host mounts (no VM)** -- sandboxed directory access with audit logging, useful for
   container-less isolation or AI agent file access control.
2. **macOS FUSE mounts** -- developers on Macs mounting directories served by a local or remote
   FS server, with the same audit and policy capabilities.
3. **Reuse as a library** -- other Rust programs embedding the server or client without depending
   on the full VMM binary or its vsock assumptions.

The FS server and FUSE client need to be extracted into a standalone library with pluggable
transports and cross-platform FUSE support.

## Non-Goals

- **General-purpose network filesystem.** This is not an NFS or CIFS replacement. It serves
  known, pre-configured mount points with tag-based routing, not arbitrary network shares.
- **Distributed filesystem.** No replication, no consensus, no multi-server coordination.
- **POSIX completeness.** Extended attributes, ACLs, file locking (`flock`/`fcntl`), and
  `mmap` are out of scope. The target workload is coding tools (editors, compilers, git, AI
  agents) which rely on basic POSIX: open/read/write/close, stat, readdir, rename, symlinks.
- **Windows support.**
- **Kernel-mode filesystem.** This is always userspace FUSE.
- **Binary distribution / CLI.** This is a library. Binaries that use it (the VMM daemon, a
  standalone mount tool) are separate crates.

## Functional Requirements

### FR-1: Cross-Platform FUSE Client
Mount a directory on the local machine backed by a remote (or local) FS server. Must work on:
- Linux (kernel 5.10+, libfuse3 / `/dev/fuse`)
- macOS (Apple Silicon, macOS 13+, via FUSE-T)

The platform difference must be invisible to the caller -- a single `mount()` API.

### FR-2: Transport-Agnostic Protocol
The wire protocol between client and server must operate over any `AsyncRead + AsyncWrite`
stream. Concrete transports supported out of the box:

| Transport | Use case |
|-----------|----------|
| vsock | VM guest ↔ host (existing motlie-vmm model) |
| Unix socket | Same-machine mounts (Linux or macOS) |
| TCP + TLS | Cross-machine mounts (macOS dev → Linux server) |

Adding a new transport must not require changes to protocol, server, or client code.

### FR-3: Pluggable Wire Encoding
The frame serialization format must be swappable. Default: bincode (fast, compact, Rust-native).
The protocol layer must be parameterized by encoding so that alternative formats (msgpack,
protobuf) can be substituted without changing the frame types or server/client logic.

### FR-4: Tag-Based Mount Routing
A single server instance serves multiple mount points, each identified by a string tag
(e.g. `workspace`, `cred-claude`). Each client connection binds to exactly one tag via a
handshake. The server maps tags to host directories.

### FR-5: Dynamic Mount Management
Mounts can be added to or removed from a running server. Adding a mount registers a new
tag → host path mapping; removing a mount deregisters it and invalidates outstanding inodes
for that tag.

### FR-6: Event Emission
Every filesystem operation that passes through the server can emit a structured event.
Events are delivered via a channel (`tokio::sync::broadcast` or similar). The library does
not persist or transport events -- that is the caller's responsibility (e.g. motlie-vmm
writes them to motlie-db).

Event emission must be optional (zero overhead when no subscriber) and must not block the
FS response path.

### FR-7: Policy Hooks
The server must support caller-provided policy functions that can intercept FS operations
before they execute. A policy function receives the operation, path, and mount tag, and
returns allow/deny. Use case: read-only credential enforcement, rate limiting credential
reads, blocking writes to specific paths.

### FR-8: Read-Only and Read-Write Mounts
Each mount tag is independently configured as read-only or read-write. Write operations
on a read-only mount return `EROFS` without reaching the host filesystem.

### FR-9: Composable Architecture
The library must support three composition modes from a single server core:

1. **Direct** -- in-process `handle_op()` calls, no serialization, no transport.
2. **vsock** -- fixed bincode over vsock, thinnest wire path, for VM use case.
3. **RPC** -- framed protocol with pluggable codec over any transport, for cross-platform mounts.

Each composite reuses the same `FsServer` core (inode table, host FS ops, events, policy).
The difference is what sits in front of `handle_op()`.

### FR-10: In-Memory Overlay with Layered Content Injection
The server must support an in-memory overlay that can intercept and replace file content
without modifying the underlying host filesystem. Requirements:

- **Multiple named layers** with explicit priority ordering. Higher-priority layers shadow
  lower ones. All layers shadow disk.
- **Per-file granularity.** Each overlay entry targets a specific path within a mount tag.
  Non-overlaid files fall through to the host filesystem.
- **Synthetic files and directories.** An overlay can inject files that do not exist on disk.
  These appear in readdir results alongside real files. Parent directories that don't exist
  on disk are created implicitly as synthetic directories (e.g., injecting `/.ssh/id_ed25519`
  implicitly creates a synthetic `/.ssh/` directory).
- **Write capture.** Writes to overlaid files update the in-memory layer, not the disk.
  Writes to non-overlaid files go to disk as normal.
- **Mutation semantics.** The overlay must define behavior for `Create`, `Unlink`, `Rename`,
  `Mkdir`, and `Rmdir` on overlay-backed paths, covering synthetic entries, shadowed files,
  and cross-boundary (overlay↔disk) operations.
- **Programmatic API** for injection at server startup or at runtime from Rust code.
- **Control socket** (Unix domain socket) for injection from the command line or external
  tooling while the server is running.

Use cases: credential injection, config overrides, hot-patching files for development,
providing synthetic content to AI agent sandboxes.

## Non-Functional Requirements

### NFR-1: Latency
Metadata operations (stat, lookup, readdir of small directories) must add < 1ms overhead
over the transport round-trip when using Unix sockets on the same machine.

### NFR-2: Throughput
Sequential read/write throughput must sustain > 500 MB/s over Unix sockets for large files,
limited by the transport and host filesystem, not by protocol overhead.

### NFR-3: Minimal Dependencies
The core module must have zero platform-specific dependencies. The server must not depend on
FUSE libraries. The client depends on `fuser` but nothing else platform-specific. Each
composite only pulls in the dependencies it needs via feature flags.

### NFR-4: Library-First
All functionality is exposed as library APIs with `pub` types and traits. No global state,
no implicit runtime. The caller provides the tokio runtime and wires components together.

### NFR-5: Testability
The server core must be testable without a real FUSE mount or any transport by calling
`handle_op()` directly. The RPC composite must be testable over in-memory duplex channels.
End-to-end FUSE tests are integration tests that can be skipped when FUSE is unavailable.

## High-Level System Design

### Composable Architecture

The library is structured around a server core with three composition layers.
All three call the same `server.handle_op()` -- the interception point for events
and policy. The composites differ only in how requests reach `handle_op()`.

```
                              ┌──────────────────────────────┐
                              │         server-core           │
                              │                               │
                              │  FsServer                     │
                              │    InodeTable (per mount tag) │
                              │    host FS ops (std::fs)      │
                              │    event emission (broadcast) │
                              │    policy enforcement         │
                              │                               │
                              │  fn handle_op(&self,          │
                              │    tag: &str,                 │
                              │    op: FsOp,                  │
                              │  ) -> FsResult                │
                              └──────┬───────────┬────────────┘
                                     │           │
            ┌────────────────────────┤           ├─────────────────────┐
            │                        │           │                     │
     ┌──────▼──────┐          ┌──────▼──────┐  ┌─▼───────────────┐    │
     │   direct     │          │   vsock      │  │   rpc            │   │
     │              │          │   composite  │  │   composite      │   │
     │ No transport │          │              │  │                  │   │
     │ No serde     │          │ bincode      │  │ Frame { id, body}│   │
     │              │          │ length-prefix│  │ Codec trait      │   │
     │ Caller calls │          │ over vsock   │  │ any transport    │   │
     │ handle_op()  │          │              │  │                  │   │
     │ directly     │          │ No Codec     │  │ RpcServer wraps  │   │
     │              │          │ trait, no    │  │   FsServer       │   │
     │ Use case:    │          │ Frame wrapper│  │ RpcClient wraps  │   │
     │ embed server │          │              │  │   ProtocolClient │   │
     │ in your own  │          │ Use case:    │  │                  │   │
     │ transport    │          │ VM guest ↔   │  │ Use case:        │   │
     │              │          │ host         │  │ cross-platform   │   │
     └──────────────┘          └──────────────┘  │ Linux host mount │   │
                                                  │ macOS mount      │   │
                                                  └──────────────────┘   │
                                                                         │
                                                  ┌──────────────────┐   │
                                                  │  client (fuser)  │◄──┘
                                                  │                  │
                                                  │ FuseClient:      │
                                                  │   fuser::        │
                                                  │   Filesystem     │
                                                  │                  │
                                                  │ Bridges FUSE ops │
                                                  │ to FsOp/FsResult │
                                                  │ via any composite│
                                                  └──────────────────┘
```

### Core Types: FsOp and FsResult

These are the types at the center of the design. Every composite converges on them.

```rust
/// A filesystem operation (request).  Core type shared by all composites.
#[derive(Debug, Serialize, Deserialize)]
pub enum FsOp {
    Lookup { parent: u64, name: String },
    Getattr { inode: u64 },
    Setattr { inode: u64, attrs: SetAttrFields },
    Readdir { inode: u64, offset: i64 },
    Open { inode: u64, flags: u32 },
    Read { inode: u64, fh: u64, offset: i64, size: u32 },
    Write { inode: u64, fh: u64, offset: i64, data: Bytes },
    Create { parent: u64, name: String, mode: u32, flags: u32 },
    Mkdir { parent: u64, name: String, mode: u32 },
    Unlink { parent: u64, name: String },
    Rmdir { parent: u64, name: String },
    Rename { parent: u64, name: String, new_parent: u64, new_name: String },
    Symlink { parent: u64, name: String, target: String },
    Readlink { inode: u64 },
    Release { inode: u64, fh: u64 },
    Fsync { inode: u64, fh: u64, datasync: bool },
    Statfs,
}

/// A filesystem result (response).  Core type shared by all composites.
#[derive(Debug, Serialize, Deserialize)]
pub enum FsResult {
    Entry { inode: u64, generation: u64, attrs: FileAttr, ttl_secs: u32 },
    Attr { attrs: FileAttr, ttl_secs: u32 },
    Data { data: Bytes },
    Written { size: u32 },
    DirEntries { entries: Vec<DirEntry> },
    Statfs { stats: FsStats },
    Symlink { target: String },
    Ok,
    Error { errno: i32 },
}
```

`Bytes` is `bytes::Bytes` -- zero-copy for bulk data during encode/decode.

### Data Flow by Composite

**Direct:**
```
caller → FsOp → server.handle_op(tag, op) → FsResult → caller
```

**vsock:**
```
fuser → FsOp → bincode → [u32 len][payload] → vsock → [u32 len][payload] → bincode → FsOp
  → server.handle_op(tag, op)
  → FsResult → bincode → vsock → bincode → FsResult → fuser
```

**RPC:**
```
fuser → FsOp → Frame { request_id, body: op } → codec.encode() → [u32 len][payload]
  → transport (unix/tcp/tls)
  → [u32 len][payload] → codec.decode() → Frame → extract FsOp
  → server.handle_op(tag, op)
  → FsResult → Frame { request_id, body: result } → codec.encode() → transport
  → codec.decode() → Frame → extract FsResult → fuser
```

The vsock path skips Frame wrapping entirely. The handshake (`Fs { tag }`) already happened at
the motlie-vmm multiplexer level, so the tag is established before the library sees the stream.
The RPC path includes its own `Hello { tag }` handshake and `request_id` for pipelining.

### Event Emission (Same for All Composites)

```
All paths converge at server.handle_op(), where:

1. Policy check:    policy.check(op, tag, path) → Allow or Err(errno)
2. Overlay check:   overlay.resolve(tag, path) → Some((layer, bytes)) or None
3. If overlaid:     return in-memory content (read) or update layer (write)
   If not overlaid: host FS op via std::fs::{read, write, stat, ...}
4. Event emit:      event_tx.try_send(FsEvent { tag, op, path, bytes })
5. Return:          FsResult
```

Events are emitted regardless of whether the content came from overlay or disk.

Events are emitted via `try_send` on a broadcast channel -- non-blocking, lossy under
backpressure (callers who care about completeness must keep up).

### In-Memory Overlay (File-Driven Memfs Layer)

The overlay is a lightweight in-memory filesystem layer inside `FsServer`. It is not a
simple path→bytes map: it is a **file-driven memfs** where injecting a file via `put()`
automatically induces the necessary parent directory hierarchy in memory. The caller only
specifies file paths; the overlay materializes synthetic directories, inodes, attrs, and
readdir behavior without anything touching host disk.

The overlay sits between policy enforcement and host filesystem access. Multiple named
layers stack by priority; resolution is top-down (highest priority first).

```
handle_op(tag, FsOp::Read { inode, .. })
    │
    ├─ policy.check() → deny? return Error { errno }
    │
    ├─ overlay layer "hotpatch" (priority 100) → hit? return in-memory content
    ├─ overlay layer "user"     (priority 10)  → hit? return in-memory content
    ├─ overlay layer "defaults" (priority 0)   → hit? return in-memory content
    │    (entries can be: Content(bytes), Whiteout, or SyntheticDir)
    │
    └─ disk (base)                             → std::fs::read()
    │
    └─ event emit
```

**Layer semantics:**
- Each layer is a named, ordered collection of `(tag, path) → Entry` mappings.
- An `Entry` is one of: `Content(Bytes)` (file data), `Whiteout` (suppresses disk file),
  or `SyntheticDir` (directory that exists only in memory).
- Layers are created with an explicit priority (`u32`). Higher wins.
- `put()` on a layer sets or replaces content for a `(tag, path)` pair and automatically
  creates `SyntheticDir` entries for any missing parent directories.
- `remove()` on a layer deletes one entry; `remove_layer()` drops all entries.
- Resolution: for a given `(tag, path)`, walk layers highest-to-lowest, return first hit.
  - `Content(bytes)` → return the in-memory content
  - `Whiteout` → return `ENOENT` (hides disk file below)
  - `SyntheticDir` → return directory inode/attrs
  - No hit → fall through to disk.

**Synthetic files and directories:**
- An overlay entry for a path that doesn't exist on disk creates a synthetic file.
  It appears in `readdir` alongside real directory entries. `lookup` and `getattr` return
  the overlay's metadata (size from content length, mtime from injection time, mode 0644).
- **Implicit synthetic directories:** When `put()` injects a file, the overlay recursively
  materializes any missing parent directories as `SyntheticDir` entries in memory. For
  example, `put("layer", "home", "/.ssh/id_ed25519", key)` implicitly creates a
  `SyntheticDir` entry for `/.ssh/` if `~/alice/.ssh/` does not exist on the host.
  The synthetic directory has mode 0755, mtime from the first child injection, and appears
  in its parent's `readdir`. `lookup("/.ssh")` returns a directory inode.
- `readdir` on a synthetic directory returns only overlay entries (there are no disk entries
  to merge, since the directory doesn't exist on disk). `readdir` on a disk directory that
  also has overlay children merges both per-file as described above.
- Synthetic directories are reference-counted: they are removed automatically when the last
  child entry in the layer is removed.

**Write behavior:**
- Writes to an overlaid path update the highest-priority layer that owns that path.
- Writes to a non-overlaid path go to disk.
- A future "capture" layer mode could intercept all writes, but this is out of scope.

**Whiteout semantics:**

A whiteout is an overlay entry that suppresses a disk file. It makes the disk file
invisible to `lookup`, `getattr`, `readdir`, and `open` — as if the file does not exist.
Whiteouts are created automatically by `unlink` on shadowed files (see mutation table).
They can also be created explicitly via the API for pre-emptive suppression.

```
resolve(tag, path):
  → Content(bytes)   → serve in-memory content
  → Whiteout         → return ENOENT (disk file hidden)
  → SyntheticDir     → return directory inode
  → None             → fall through to disk
```

**Mutation semantics:**

Operations on overlay-backed paths follow filesystem-consistent behavior. The key
principles: (1) deleting a visible file makes it disappear — it never resurrects hidden
content, (2) renames across the overlay↔disk boundary are handled transparently to
support editor atomic-save flows, (3) the overlay behaves like a real filesystem layer,
not a cache.

| Operation | Target | Behavior |
|-----------|--------|----------|
| `Unlink` | Synthetic file | Remove overlay entry. File vanishes. |
| `Unlink` | Shadowed file (overlay over disk) | Replace overlay `Content` entry with `Whiteout`. File disappears; disk file stays hidden. |
| `Unlink` | Whiteout | Return `ENOENT` (already deleted). |
| `Unlink` | Disk-only file | Normal `std::fs::remove_file()`. |
| `Create` | Under synthetic directory | Create `Content` entry in overlay (in-memory). |
| `Create` | Under disk directory (no overlay children) | Normal `std::fs::create()` on disk. |
| `Create` | Under disk directory (has overlay children) | Create `Content` entry in overlay (in-memory). Keeps overlay-managed paths consistent. |
| `Mkdir` | Under synthetic parent | Create `SyntheticDir` entry in overlay. |
| `Mkdir` | Under disk parent | Normal `std::fs::create_dir()` on disk. |
| `Rmdir` | Synthetic directory (empty) | Remove `SyntheticDir` from overlay. |
| `Rmdir` | Disk directory | Normal `std::fs::remove_dir()`. |
| `Rename` | Overlay → overlay | Rename within overlay (in-memory). |
| `Rename` | Disk → disk | Normal `std::fs::rename()`. |
| `Rename` | Disk → overlay target | Read disk source content, update overlay entry at target, `std::fs::remove_file()` disk source. Editor atomic-save compatible. |
| `Rename` | Overlay source → disk target | Write overlay content to disk at target path, remove overlay entry for source. |
| `Setattr` | Overlaid/synthetic file | Update in-memory attrs. |
| `Setattr` | Disk file | Normal `std::fs` operation. |

**Why transparent cross-boundary rename (not EXDEV):**

The stated target workload is coding tools (editors, compilers, git). Editors perform
atomic saves by writing to a temp file and renaming over the target:

```
vim writes /root/.ssh/config.tmp     (new file — created in overlay because .ssh has overlay children)
vim renames config.tmp → config      (overlay → overlay rename — works)
```

If the temp file is created on disk (e.g., in a non-overlay directory), the rename is
disk→overlay. Rather than returning `EXDEV` (which breaks the editor flow), the server
handles this transparently: read the disk source, write into the overlay target, delete
the disk source. From the editor's perspective, the rename succeeded normally.

## Crate Structure

```
libs/vfs/
├── Cargo.toml
├── docs/
│   └── DESIGN.md                 # this document
└── src/
    ├── lib.rs                    # re-exports, feature-gated module visibility
    │
    ├── core/
    │   ├── mod.rs
    │   ├── op.rs                 # FsOp, FsResult, FileAttr, DirEntry, FsStats
    │   ├── server.rs             # FsServer, FsServerBuilder, handle_op()
    │   ├── inode.rs              # InodeTable: inode ↔ host path mapping
    │   ├── overlay.rs            # MemOverlay: layered in-memory content injection
    │   ├── event.rs              # FsEvent, FsOp (event enum), EventSender
    │   └── policy.rs             # PolicyFn trait, AllowAll default
    │
    ├── control/                  # feature = "control"
    │   ├── mod.rs
    │   ├── socket.rs             # ControlSocket: Unix domain socket listener
    │   └── protocol.rs           # Line protocol: LAYER, PUT, RM, GET, LS, etc.
    │
    ├── vsock/                    # feature = "vsock"
    │   ├── mod.rs
    │   ├── handler.rs            # VsockConnectionHandler: bincode FsOp/FsResult over stream
    │   └── mount.rs              # VsockFuseMount: fuser::Filesystem over vsock+bincode
    │
    ├── rpc/                      # feature = "rpc"
    │   ├── mod.rs
    │   ├── frame.rs              # Frame, FrameBody (wraps FsOp/FsResult with request_id)
    │   ├── codec.rs              # Codec trait
    │   ├── io.rs                 # read_frame / write_frame (length-prefixed)
    │   ├── server.rs             # RpcServer: wraps FsServer, frame decode → handle_op()
    │   └── client.rs             # RpcClient: frame encode/decode + handshake
    │
    ├── codec/                    # wire encoding implementations
    │   ├── mod.rs
    │   └── bincode.rs            # feature = "bincode-codec"
    │
    └── client/                   # feature = "client"
        ├── mod.rs
        └── fuse.rs               # FuseClient: fuser::Filesystem backed by any composite
```

### Feature Flags

```toml
[features]
default = ["server-core", "bincode-codec"]

# Always available -- core types, FsServer, MemOverlay
server-core = []

# Composites
vsock = ["server-core", "bincode-codec", "dep:tokio-vsock"]
rpc = ["server-core"]

# FUSE client (used with either vsock or rpc composite)
client = ["dep:fuser"]

# Control socket for overlay management
control = ["server-core"]

# Wire encodings
bincode-codec = ["dep:bincode"]
# msgpack-codec = ["dep:rmp-serde"]   # future

# Transport extras
tls = ["rpc", "dep:tokio-rustls", "dep:rustls"]
```

**Dependency profiles for each consumer:**

| Consumer | Features | What compiles |
|----------|----------|---------------|
| motlie-vmm host | `vsock` | core + overlay + vsock handler + bincode |
| motlie-vmm guest | `vsock, client` | core + vsock mount + fuser + bincode |
| Standalone FS server | `rpc, bincode-codec, control` | core + overlay + rpc server + control socket + bincode |
| Linux/macOS mount client | `rpc, client, bincode-codec` | core + rpc client + fuser + bincode |
| Remote mount + TLS | `rpc, client, bincode-codec, tls` | above + rustls |
| Custom embedding | `server-core` | core + overlay, caller provides own transport |
| Tests (no FUSE) | `rpc, bincode-codec` | core + overlay + rpc, no fuser |

## API Design

### Server Core

```rust
/// Core filesystem server.  All composites call handle_op().
pub struct FsServer { /* ... */ }

impl FsServer {
    pub fn builder() -> FsServerBuilder;

    /// Direct operation dispatch.  The single entry point that all composites use.
    /// Tag must match a registered mount.  Returns FsResult::Error { errno: ENOENT }
    /// if the tag is unknown.
    pub fn handle_op(&self, tag: &str, op: FsOp) -> FsResult;

    /// Register a new mount tag.  Can be called while serving.
    pub async fn add_mount(&self, tag: &str, host_path: PathBuf, read_only: bool) -> Result<()>;

    /// Remove a mount tag.  Drops inode table; in-flight ops get ENOENT.
    pub async fn remove_mount(&self, tag: &str) -> Result<()>;

    /// Subscribe to filesystem events (if event emission is enabled).
    pub fn subscribe_events(&self) -> Option<broadcast::Receiver<FsEvent>>;
}

pub struct FsServerBuilder { /* ... */ }

impl FsServerBuilder {
    pub fn mount(self, tag: &str, host_path: PathBuf, read_only: bool) -> Self;
    pub fn events(self, capacity: usize) -> Self;
    pub fn policy(self, policy: impl PolicyFn) -> Self;
    pub fn build(self) -> Result<FsServer>;
}
```

### Event Types

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FsEvent {
    pub timestamp: SystemTime,
    pub tag: String,
    pub op: FsOp,
    pub path: String,
    pub bytes: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FsOp {
    Lookup, Getattr, Open, Read, Write, Create,
    Mkdir, Unlink, Rmdir, Rename, Symlink, Readlink,
}
```

The motlie-vmm daemon wraps these with VM identity and credential classification
(using its own `AuditedFsServer` wrapper). The library emits raw FS events; the caller
adds domain context.

### Policy Trait

```rust
pub trait PolicyFn: Send + Sync + 'static {
    /// Check whether an operation is allowed.
    /// Return Ok(()) to allow, Err(errno) to deny.
    /// Called for every FsOp including individual Readdir entries.
    fn check(&self, op: FsOp, tag: &str, path: &str) -> Result<(), i32>;
}

/// Default: allow everything.
pub struct AllowAll;
impl PolicyFn for AllowAll {
    fn check(&self, _op: FsOp, _tag: &str, _path: &str) -> Result<(), i32> {
        Ok(())
    }
}
```

### In-Memory Overlay API

The overlay is a file-driven memfs layer, part of server-core (always available when
FsServer is compiled). No additional feature flag required.

Callers inject files; the overlay materializes synthetic parent directories automatically.
The data model tracks three entry types: `Content(Bytes)`, `Whiteout`, and `SyntheticDir`.

```rust
/// File-driven in-memory filesystem layer.
/// Injecting a file automatically materializes synthetic parent directories.
/// Accessed via server.overlay().
pub struct MemOverlay { /* ... */ }

/// What an overlay entry contains.
#[derive(Clone, Debug)]
pub enum OverlayEntryKind {
    /// File content (injected via put() or captured from writes).
    Content(Bytes),
    /// Suppresses a disk file — makes it invisible (created by unlink on shadowed files).
    Whiteout,
    /// Directory that exists only in memory (created implicitly by put() for missing parents).
    SyntheticDir,
}

impl MemOverlay {
    // --- Layer management ---

    /// Create or update a named layer with a priority.
    /// Higher priority layers shadow lower ones.
    pub fn put_layer(&self, name: &str, priority: u32) -> Result<()>;

    /// Remove a layer and all its entries (including whiteouts and synthetic dirs).
    pub fn remove_layer(&self, name: &str) -> Result<()>;

    /// List all layers, ordered by priority (highest first).
    pub fn layers(&self) -> Vec<LayerInfo>;

    // --- Content management (within a layer) ---

    /// Inject a file into the overlay.  Automatically creates SyntheticDir entries
    /// for any missing parent directories in this layer.
    /// If the path already has an entry (Content, Whiteout, or SyntheticDir), replaces it.
    pub fn put(&self, layer: &str, tag: &str, path: &str, content: Bytes) -> Result<()>;

    /// Create a whiteout entry that suppresses a disk file.
    pub fn whiteout(&self, layer: &str, tag: &str, path: &str) -> Result<()>;

    /// Remove one overlay entry.  For Content/Whiteout: file falls back to disk
    /// (or disappears if synthetic).  Synthetic parent dirs are removed if they
    /// have no remaining children.
    pub fn remove(&self, layer: &str, tag: &str, path: &str) -> Result<()>;

    /// Read the content stored in a specific layer (for debugging/inspection).
    pub fn get(&self, layer: &str, tag: &str, path: &str) -> Option<Bytes>;

    /// List all entries in a layer for a mount tag.
    pub fn list_layer(&self, layer: &str, tag: &str) -> Vec<OverlayEntry>;

    // --- Resolved view (what handle_op() sees) ---

    /// Resolve a path: walk layers highest-to-lowest, return first hit.
    /// Returns the entry kind. None means fall through to disk.
    pub fn resolve(&self, tag: &str, path: &str) -> Option<(String, OverlayEntryKind)>;

    /// List all effective overlays for a tag (one entry per path, from the
    /// highest-priority layer that owns it).
    pub fn list_effective(&self, tag: &str) -> Vec<EffectiveEntry>;
}

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub priority: u32,
    pub entry_count: usize,
}

#[derive(Debug, Clone)]
pub struct OverlayEntry {
    pub path: String,
    pub size: usize,
    pub injected_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct EffectiveEntry {
    pub path: String,
    pub layer: String,
    pub size: usize,
}
```

Wired into FsServer:

```rust
impl FsServerBuilder {
    /// Enable the in-memory overlay (default: disabled, handle_op goes straight to disk).
    pub fn overlay(self, enabled: bool) -> Self;
}

impl FsServer {
    /// Access the overlay.  Returns None if overlay was not enabled in the builder.
    pub fn overlay(&self) -> Option<&MemOverlay>;
}
```

### Control Socket (Unix Domain Socket)

A lightweight control interface for managing the overlay from the command line or external
tooling at runtime. Feature-gated behind `control`.

The control socket speaks a simple text-based line protocol over a Unix domain socket.
Binary file content is transferred with a length prefix after the command line.

**Library API:**

```rust
/// Listens on a Unix domain socket and dispatches commands to FsServer's overlay.
pub struct ControlSocket { /* ... */ }

impl ControlSocket {
    pub fn new(server: &FsServer) -> Self;

    /// Listen and serve control commands.  Blocks until the socket is closed.
    pub async fn listen(&self, path: &Path) -> Result<()>;
}
```

**Wire protocol:**

Each command is a single line (`\n`-terminated). Commands that carry content include a
byte length; the content follows immediately after the newline.

```
Commands:

  LAYER <name> <priority>\n           Create/update layer with priority (u32)
  RMLAYER <name>\n                    Remove layer and all its entries
  LAYERS\n                            List all layers

  PUT <layer> <tag> <path> <len>\n    Set overlay content (len bytes follow)
  <len bytes of content>
  RM <layer> <tag> <path>\n           Remove one overlay entry
  GET <layer> <tag> <path>\n          Read overlay content from specific layer
  LS <tag>\n                          List effective overlays for a tag
  LSLAYER <layer> <tag>\n             List entries in a specific layer for a tag

Responses:

  OK\n                                Success (for LAYER, RMLAYER, PUT, RM)

  LAYERS\n                            Response to LAYERS command
  <name> <priority> <entry_count>\n   One line per layer (highest priority first)
  \n                                  Empty line terminates list

  DATA <len>\n                        Response to GET
  <len bytes of content>

  ENTRIES\n                           Response to LS or LSLAYER
  <path> <layer> <size>\n             One line per entry (LS)
  <path> <size>\n                     One line per entry (LSLAYER)
  \n                                  Empty line terminates list

  ERR <message>\n                     Error
```

**CLI usage examples:**

```bash
# Inject a config file into the "defaults" layer at startup
echo 'KEY=value' | motlie-vfs-ctl put defaults workspace /.env

# Inject from a file
motlie-vfs-ctl put defaults workspace /config.toml < config.toml

# Create a high-priority hotpatch layer and override a file
motlie-vfs-ctl layer hotpatch 100
motlie-vfs-ctl put hotpatch workspace /src/config.rs < patched_config.rs

# List what's overlaid for the workspace mount
motlie-vfs-ctl ls workspace
# → /config.toml hotpatch 4821
# → /.env defaults 42

# Read back what's in the overlay
motlie-vfs-ctl get defaults workspace /.env
# → KEY=value

# Remove the hotpatch
motlie-vfs-ctl rmlayer hotpatch

# List layers
motlie-vfs-ctl layers
# → defaults 0 2
# → alice 10 1
```

The `motlie-vfs-ctl` CLI tool is a separate binary (not part of the library). It's a thin
wrapper that connects to the Unix socket and speaks the line protocol. It can be implemented
in ~100 lines. For simple cases, `socat` works directly:

```bash
echo -e "LAYERS" | socat - UNIX-CONNECT:/tmp/motlie-vfs-ctl.sock

# PUT with content (requires printf for binary-safe length prefix)
content="KEY=value"
printf "PUT defaults workspace /.env %d\n%s" ${#content} "$content" \
  | socat - UNIX-CONNECT:/tmp/motlie-vfs-ctl.sock
```

### Overlay Control: Frontend Architecture

The `MemOverlay` is a Rust API on `FsServer`. The control socket is one frontend to it,
not the only one. Any code with access to `server.overlay()` can call `put`/`remove`/
`put_layer`. This makes the overlay composable with arbitrary control frontends.

```
MemOverlay (Rust API on FsServer)       ← the core, always available
    ↑           ↑          ↑        ↑
    │           │          │        │
ControlSocket  HTTP/REST  gRPC     In-process
(Unix socket)  (axum)     (tonic)  (direct Rust calls)
  provided     caller's   caller's   caller's
  by library   crate      crate      code
    │           │          │        │
CLI tool       curl       client   motlie-vmm daemon
```

The library provides `MemOverlay` (always) + `ControlSocket` (feature-gated). Network
frontends (HTTP, gRPC) are the caller's responsibility — they import `motlie-vfs` with
`server-core`, obtain `server.overlay()`, and wrap it in their own transport. No library
changes needed.

**Example: HTTP/REST frontend (separate crate, not part of motlie-vfs):**

```rust
use motlie_vfs::core::FsServer;
use axum::{Router, extract::Path, body::Bytes, http::StatusCode};

let server: Arc<FsServer> = /* ... */;

let app = Router::new()
    .route("/overlay/:layer/:tag/*path", put({
        let server = server.clone();
        move |Path((layer, tag, path)): Path<(String, String, String)>,
              body: Bytes| async move {
            server.overlay().unwrap().put(&layer, &tag, &path, body)?;
            Ok::<_, _>(StatusCode::OK)
        }
    }))
    .route("/overlay/:layer/:tag/*path", delete({
        let server = server.clone();
        move |Path((layer, tag, path)): Path<(String, String, String)>| async move {
            server.overlay().unwrap().remove(&layer, &tag, &path)?;
            Ok::<_, _>(StatusCode::OK)
        }
    }))
    .route("/overlay/:layer/:tag/*path", get({
        let server = server.clone();
        move |Path((layer, tag, path)): Path<(String, String, String)>| async move {
            match server.overlay().unwrap().get(&layer, &tag, &path) {
                Some(data) => Ok(data),
                None => Err(StatusCode::NOT_FOUND),
            }
        }
    }));

axum::serve(TcpListener::bind("0.0.0.0:9001").await?, app).await?;
```

This turns motlie-vfs into a network file server with a REST API for content injection:

```bash
# Remote client injects an SSH key over the network
curl -X PUT https://fileserver:9001/overlay/credentials/home/.ssh/id_deploy \
  --data-binary @deploy_key

# Inject API tokens
curl -X PUT https://fileserver:9001/overlay/credentials/home/.env \
  --data-binary @- <<< "ANTHROPIC_API_KEY=sk-ant-..."

# Read back
curl https://fileserver:9001/overlay/credentials/home/.ssh/id_deploy

# Remove
curl -X DELETE https://fileserver:9001/overlay/credentials/home/.ssh/id_deploy
```

**Dynamic credential injection while mounted (all frontends):**

Because `handle_op()` checks the overlay on every operation with no caching, a `PUT`
followed by a `read()` in the guest sees the new content immediately. This enables
dynamic, mid-session credential management:

```bash
# Mid-session: inject a deploy key (via control socket, HTTP, or in-process)
motlie-vfs-ctl put credentials home /.ssh/id_deploy < deploy_key
# OR: curl -X PUT https://server:9001/overlay/credentials/home/.ssh/id_deploy --data-binary @deploy_key
# OR: server.overlay().put("credentials", "home", "/.ssh/id_deploy", key_bytes)

# Guest immediately sees it — no restart, no remount
ssh -i ~/.ssh/id_deploy git@github.com   # works

# Shadow an existing key (disk version hidden, overlay version served)
motlie-vfs-ctl put credentials home /.ssh/id_ecdsa < ephemeral_key
# Guest reads /root/.ssh/id_ecdsa → gets ephemeral_key, not the on-disk version

# Remove — original disk file reappears, synthetic files vanish
motlie-vfs-ctl rm credentials home /.ssh/id_deploy    # gone
motlie-vfs-ctl rm credentials home /.ssh/id_ecdsa     # disk version reappears
```

**motlie-vmm integration:**

For the VM use case, the VMM daemon calls `server.overlay()` directly from Rust — no control
socket or HTTP needed. The VMM's `ControlMsg::AddMount` handler can inject overlay content
as part of VM setup. The control socket and HTTP frontends are for standalone server
deployments where external tooling needs to manage overlays at runtime.

### vsock Composite

Thin layer: length-prefixed bincode of `FsOp`/`FsResult` directly over a stream.
No `Frame` wrapper, no `Codec` trait, no handshake (the motlie-vmm multiplexer
handles the `Fs { tag }` handshake before handing off the stream).

```rust
/// Host side: serves a single vsock connection for a known tag.
pub struct VsockConnectionHandler { /* ... */ }

impl VsockConnectionHandler {
    /// Create a handler bound to a specific tag.
    /// The tag is already established by the caller (vmm multiplexer handshake).
    pub fn new(server: &FsServer, tag: &str) -> Self;

    /// Serve the connection: read FsOp, call handle_op(), write FsResult.
    /// Loops until the stream closes.
    pub async fn serve<S>(&self, stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send;
}
```

```rust
/// Guest side: fuser::Filesystem backed by bincode over a stream.
pub struct VsockFuseMount { /* ... */ }

impl VsockFuseMount {
    /// Create a FUSE mount backed by the given stream.
    /// The tag is already established by the caller (guest agent handshake).
    pub fn new<S>(stream: S, tag: &str) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static;
}

// VsockFuseMount implements fuser::Filesystem.
```

### RPC Composite

Full-featured protocol layer: `Frame` wrapping, pluggable `Codec`, pipelining support,
built-in handshake. For cross-platform use where client and server may be different
machines/OSes.

```rust
/// Frame wraps FsOp/FsResult with request_id and handshake.
#[derive(Serialize, Deserialize, Debug)]
pub struct Frame {
    pub request_id: u64,
    pub body: FrameBody,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FrameBody {
    // Handshake
    Hello { version: u32, tag: String },
    HelloOk { version: u32 },
    HelloErr { message: String },

    // Wraps core types
    Request(FsOp),
    Response(FsResult),
}
```

```rust
/// Pluggable wire encoding.
pub trait Codec: Send + Sync + 'static {
    fn encode(&self, frame: &Frame, buf: &mut Vec<u8>) -> Result<()>;
    fn decode(&self, buf: &[u8]) -> Result<Frame>;
}

pub struct BincodeCodec;
impl Codec for BincodeCodec { /* ... */ }
```

```rust
/// Host side: wraps FsServer with frame protocol.
pub struct RpcServer { /* ... */ }

impl RpcServer {
    pub fn new<C: Codec>(server: FsServer, codec: C) -> Self;

    /// Serve a single client connection.
    /// Handles Hello handshake, then frame decode → handle_op() → frame encode loop.
    pub async fn serve<S>(&self, stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send;
}
```

```rust
/// Client side: frame-level protocol client (no FUSE).
pub struct RpcClient { /* ... */ }

impl RpcClient {
    /// Connect and perform Hello handshake.
    pub async fn connect<S, C>(stream: S, codec: C, tag: &str) -> Result<Self>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
        C: Codec;

    /// Send FsOp, receive FsResult.
    pub async fn request(&self, op: FsOp) -> Result<FsResult>;

    /// Convert to a fuser::Filesystem for mounting.
    /// Consumes self.
    pub fn into_fuse(self) -> FuseClient;
}
```

### Frame I/O

Wire format for both vsock and RPC composites: `[u32 big-endian length][payload]`.

For vsock composite, payload is `bincode(FsOp)` or `bincode(FsResult)`.
For RPC composite, payload is `codec.encode(Frame)`.

```rust
/// Read/write length-prefixed payloads over any async stream.
pub async fn write_msg<T: AsyncWrite + Unpin>(
    stream: &mut T,
    payload: &[u8],
) -> Result<()>;

pub async fn read_msg<T: AsyncRead + Unpin>(
    stream: &mut T,
    max_size: usize,
) -> Result<Vec<u8>>;
```

The framing layer is shared; the interpretation of the payload differs by composite.

## Composition Patterns (Usage Examples)

### Pattern 1: Direct -- Embed Server in Your Own Transport

The caller owns the transport and encoding. The library provides only the FS engine.

```rust
use motlie_vfs::core::{FsServer, FsOp, FsResult};

let server = FsServer::builder()
    .mount("workspace", "/home/alice/projects".into(), false)
    .events(4096)
    .build()?;

// Subscribe to events in a background task
let mut events = server.subscribe_events().unwrap();
tokio::spawn(async move {
    while let Ok(ev) = events.recv().await {
        println!("{:?}", ev);
    }
});

// You decode requests however you want, then call handle_op:
let result = server.handle_op("workspace", FsOp::Lookup {
    parent: 1,
    name: "src".into(),
});
match result {
    FsResult::Entry { inode, attrs, .. } => { /* route response back */ }
    FsResult::Error { errno } => { /* handle error */ }
    _ => {}
}
```

### Pattern 2: vsock -- motlie-vmm Host Side

The VMM daemon handles the vsock multiplexer and dispatches `Fs { tag }` connections
to the library.  The library never sees the `HandshakeMsg` enum -- it receives a stream
that is already bound to a tag.

```rust
use motlie_vfs::core::FsServer;
use motlie_vfs::vsock::VsockConnectionHandler;

let server = FsServer::builder()
    .mount("workspace", projects_dir.join(&username), false)
    .mount("cred-claude", creds_dir.join(&username).join(".claude"), false)
    .events(4096)
    .policy(ReadOnlyCredentials::new())
    .build()?;

// In the vsock multiplexed listener, after HandshakeMsg dispatch:
match handshake {
    HandshakeMsg::Fs { tag } => {
        let handler = VsockConnectionHandler::new(&server, &tag);
        handler.serve(vsock_stream).await;
    }
    HandshakeMsg::BinaryRequest => { /* ... handled by vmm, not the lib */ }
    HandshakeMsg::Control => { /* ... handled by vmm, not the lib */ }
}
```

### Pattern 3: vsock -- motlie-vmm Guest Side

The guest agent opens a vsock connection, performs the VMM-level handshake, then
hands the stream to the library for FUSE mounting.

```rust
use motlie_vfs::vsock::VsockFuseMount;
use tokio_vsock::VsockStream;

// Guest agent: open vsock, do the vmm-level handshake
let mut stream = VsockStream::connect(2, 5000).await?;
send_handshake(&mut stream, &HandshakeMsg::Fs { tag: "workspace".into() }).await?;

// Hand the established stream to the library for FUSE mounting
let mount = VsockFuseMount::new(stream, "workspace");
fuser::mount2(mount, "/workspace", &[MountOption::AutoUnmount])?;
```

### Pattern 4: RPC Server -- Standalone Linux/macOS FS Server

A standalone binary (separate crate) that uses the library to serve mounts over
a Unix socket or TCP.

```rust
use motlie_vfs::core::FsServer;
use motlie_vfs::rpc::RpcServer;
use motlie_vfs::codec::BincodeCodec;
use tokio::net::UnixListener;

let server = FsServer::builder()
    .mount("workspace", "/home/alice/projects".into(), false)
    .mount("data", "/data/shared".into(), true)
    .events(4096)
    .policy(ReadOnlyCredentials::new())
    .build()?;

let rpc = RpcServer::new(server, BincodeCodec);

// Caller owns the listener
let listener = UnixListener::bind("/tmp/motlie-vfs.sock")?;
loop {
    let (stream, _) = listener.accept().await?;
    let rpc = rpc.clone();
    tokio::spawn(async move {
        // Handles Hello handshake → frame loop → handle_op()
        rpc.serve(stream).await
    });
}
```

### Pattern 5: RPC Client -- Mount on Linux or macOS

```rust
use motlie_vfs::rpc::RpcClient;
use motlie_vfs::codec::BincodeCodec;
use tokio::net::UnixStream;

// Connect to server
let stream = UnixStream::connect("/tmp/motlie-vfs.sock").await?;

// RpcClient handles Hello handshake and frame encode/decode
let client = RpcClient::connect(stream, BincodeCodec, "workspace").await?;

// Convert to fuser::Filesystem and mount
// Identical on Linux and macOS (FUSE-T provides libfuse compat on macOS)
fuser::mount2(client.into_fuse(), "/mnt/workspace", &[
    MountOption::AutoUnmount,
])?;
```

### Pattern 6: RPC Client -- Remote Mount over TCP+TLS

```rust
use motlie_vfs::rpc::RpcClient;
use motlie_vfs::codec::BincodeCodec;
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

let tcp = TcpStream::connect("build-server:9000").await?;
let tls = TlsConnector::from(tls_config);
let stream = tls.connect(server_name, tcp).await?;

let client = RpcClient::connect(stream, BincodeCodec, "workspace").await?;
fuser::mount2(client.into_fuse(), "/Volumes/workspace", &[
    MountOption::AutoUnmount,
])?;
```

### Pattern 7: Testing -- No FUSE, No Transport

```rust
use motlie_vfs::core::{FsServer, FsOp, FsResult};

#[tokio::test]
async fn test_read_write() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), b"world").unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .build()
        .unwrap();

    // Direct: no transport, no serde, no FUSE
    let entry = server.handle_op("test", FsOp::Lookup {
        parent: 1,
        name: "hello.txt".into(),
    });
    let inode = match entry {
        FsResult::Entry { inode, .. } => inode,
        other => panic!("expected Entry, got {:?}", other),
    };

    let open = server.handle_op("test", FsOp::Open { inode, flags: 0 });
    let fh = match open {
        FsResult::Entry { .. } => { /* extract fh */ 0 },
        other => panic!("expected open result, got {:?}", other),
    };

    let data = server.handle_op("test", FsOp::Read {
        inode, fh, offset: 0, size: 4096,
    });
    match data {
        FsResult::Data { data } => assert_eq!(&data[..], b"world"),
        other => panic!("expected Data, got {:?}", other),
    }
}
```

### Pattern 8: Testing -- RPC Over In-Memory Duplex (No FUSE)

```rust
use motlie_vfs::core::FsServer;
use motlie_vfs::rpc::{RpcServer, RpcClient};
use motlie_vfs::codec::BincodeCodec;

#[tokio::test]
async fn test_rpc_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), b"world").unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .build()
        .unwrap();

    let rpc_server = RpcServer::new(server, BincodeCodec);

    // In-memory transport -- no Unix socket, no FUSE
    let (client_stream, server_stream) = tokio::io::duplex(64 * 1024);

    tokio::spawn(async move { rpc_server.serve(server_stream).await });

    let client = RpcClient::connect(client_stream, BincodeCodec, "test").await.unwrap();

    let result = client.request(FsOp::Lookup {
        parent: 1,
        name: "hello.txt".into(),
    }).await.unwrap();

    assert!(matches!(result, FsResult::Entry { .. }));
}
```

### Pattern 9: Overlay -- Startup Injection + Runtime Control Socket

```rust
use motlie_vfs::core::FsServer;
use motlie_vfs::rpc::RpcServer;
use motlie_vfs::control::ControlSocket;
use motlie_vfs::codec::BincodeCodec;

let server = FsServer::builder()
    .mount("workspace", "/home/alice/projects".into(), false)
    .overlay(true)
    .events(4096)
    .build()?;

// Startup: inject defaults layer with base config
let overlay = server.overlay().unwrap();
overlay.put_layer("defaults", 0)?;
overlay.put("defaults", "workspace", "/.env", Bytes::from("API_KEY=sk-test-123"))?;
overlay.put("defaults", "workspace", "/config.toml", config_bytes)?;

// RPC server for FUSE clients
let rpc = RpcServer::new(server.clone(), BincodeCodec);
let listener = UnixListener::bind("/tmp/motlie-vfs.sock")?;
tokio::spawn(async move {
    loop {
        let (stream, _) = listener.accept().await.unwrap();
        let rpc = rpc.clone();
        tokio::spawn(async move { rpc.serve(stream).await });
    }
});

// Control socket for runtime overlay management
let ctl = ControlSocket::new(&server);
ctl.listen("/tmp/motlie-vfs-ctl.sock".as_ref()).await?;
// Now external tools can: motlie-vfs-ctl put hotpatch workspace /config.toml < new_config.toml
```

### Pattern 10: Overlay -- Direct Testing

```rust
use motlie_vfs::core::{FsServer, FsOp, FsResult};

#[tokio::test]
async fn test_overlay_shadows_disk() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.toml"), b"disk content").unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .overlay(true)
        .build()
        .unwrap();

    // Without overlay: read returns disk content
    let inode = lookup(&server, "test", "config.toml");
    let data = read_file(&server, "test", inode);
    assert_eq!(&data[..], b"disk content");

    // Inject overlay
    let overlay = server.overlay().unwrap();
    overlay.put_layer("patch", 10).unwrap();
    overlay.put("patch", "test", "/config.toml", Bytes::from("overlaid")).unwrap();

    // Same read now returns overlay content
    let data = read_file(&server, "test", inode);
    assert_eq!(&data[..], b"overlaid");

    // Remove overlay: falls back to disk
    overlay.remove("patch", "test", "/config.toml").unwrap();
    let data = read_file(&server, "test", inode);
    assert_eq!(&data[..], b"disk content");
}

#[tokio::test]
async fn test_overlay_synthetic_file() {
    let dir = tempfile::tempdir().unwrap();

    let server = FsServer::builder()
        .mount("test", dir.path().to_path_buf(), false)
        .overlay(true)
        .build()
        .unwrap();

    let overlay = server.overlay().unwrap();
    overlay.put_layer("inject", 0).unwrap();
    overlay.put("inject", "test", "/.env", Bytes::from("SECRET=abc")).unwrap();

    // Synthetic file: doesn't exist on disk but is visible via lookup + readdir
    let entry = server.handle_op("test", FsOp::Lookup {
        parent: 1,
        name: ".env".into(),
    });
    assert!(matches!(entry, FsResult::Entry { .. }));

    // Also appears in directory listing
    let entries = server.handle_op("test", FsOp::Readdir { inode: 1, offset: 0 });
    // entries includes .env alongside any real files
}
```

## Inode Management

The server maintains a per-mount bidirectional mapping between inodes (u64) and host paths.
Inode 1 is always the root of the mount. Inodes are allocated on `lookup` and cached.

```
InodeTable:
  inode → (host_path, refcount, attrs_cache)
  host_path → inode
```

Inodes are scoped per mount tag -- two mounts can independently use the same inode numbers.
This is safe because each client connection is bound to exactly one tag, and the FUSE kernel
module per mount point has its own inode namespace.

When a mount is removed via `remove_mount()`, its `InodeTable` is dropped. Any in-flight
connection for that tag will receive `FsResult::Error { errno: ENOENT }` on subsequent
requests.

## macOS: FUSE-T Strategy

macOS FUSE support uses **FUSE-T** (not macFUSE):

| Property | FUSE-T | macFUSE |
|----------|--------|---------|
| Mechanism | Userspace NFS loopback | Kernel extension (kext) |
| SIP | No changes needed | Must lower security |
| Install | `brew install fuse-t` | PKG + reboot |
| `fuser` compat | Yes (provides `libfuse.dylib`) | Yes |
| Future-proof | Yes (kexts deprecated by Apple) | Deprecated path |
| Perf overhead | ~10-20% from NFS translation | Native kext speed |

The `fuser` crate links against `libfuse.dylib` at runtime. On macOS with FUSE-T installed,
this resolves to FUSE-T's compatibility library. No conditional compilation needed in our
code -- `fuser::mount2()` works identically on both platforms.

**Build-time check**: The client feature should include a build script that verifies FUSE
headers are available (Linux: `libfuse3-dev`, macOS: FUSE-T) and emits a clear error message
if not.

## Mapping to motlie-vmm Design Doc

This section documents how motlie-vfs components map to the motlie-vmm architecture
(sections 10-12 of `motlie/docs/motlie-vmm.md`).

### What the Library Extracts

| motlie-vmm concept | motlie-vfs component | Notes |
|---|---|---|
| vsock FS server (~800 lines, §10) | `FsServer` (core) | Same job: tag→host_path routing, inode table, host FS ops |
| `fs_loop(stream, vm, &tag)` in §10 | `VsockConnectionHandler::serve()` | Extracted + parameterized over stream type |
| `FsServer::add_mount()` / `remove_mount()` in §11 | `FsServer::add_mount()` / `remove_mount()` | Identical interface |
| Guest agent FUSE↔vsock bridge (~700 lines, §10-11) | `VsockFuseMount` / `FuseClient` | Same job, transport-agnostic |
| `VsockFuse::new(HOST_CID, VMM_PORT, &tag, read_only)` in §11 | `VsockFuseMount::new(stream, tag)` | Takes established stream, not raw vsock params |
| Inline policy checks in §12 (`if is_cred { ... }`) | `PolicyFn` trait | Formalized into trait |
| `AuditedFsServer` wrapping `FsServer` in §12 | `FsServer` with `.events()` builder | Events built into core, domain context added by caller |

### What Stays in motlie-vmm

| motlie-vmm concept | Why not extracted |
|---|---|
| `HandshakeMsg` enum (`BinaryRequest`, `Control`, `Fs`) | Multiplexer-level routing, not an FS concern |
| `ControlMsg` enum (`Ready`, `AddMount`, `RemoveMount`, `Shutdown`) | VM lifecycle control; calls into `FsServer::add_mount()` |
| `AuditedFsServer` domain enrichment (vm_name, credential classification) | VMM-specific; subscribes to `FsEvent` and adds context |
| Bootstrap binary delivery (`BinaryRequest`) | VM-specific bootstrap |
| Guest agent `main()` loop (§11) | Orchestration: control channel + spawning mounts |

## VMM Guest Integration: vsock + Overlay

This section documents how motlie-vfs integrates with the motlie-vmm guest agent over
vsock, including the full VM boot lifecycle and how the overlay enables selective in-memory
content injection for specific paths within a pass-through mount.

### VM Boot Lifecycle (from Firecracker to FUSE mounts)

The guest agent is the only custom binary inside the VM. It arrives over vsock at boot,
never touches disk, and uses motlie-vfs for all FUSE operations. The boot sequence:

```
Host (motlie-vmm daemon)                     Guest (Firecracker VM)
─────────────────────────                    ───────────────────────
ssh alice@:2222 arrives
  │
ensure_vm("alice")
  │
  ├─ create overlay.ext4
  │   inject: CA keys, principals,
  │   mounts.json, env
  │
  ├─ spawn passt (memfd, in netns)
  ├─ spawn firecracker (memfd, in netns)
  │   kernel cmdline:
  │     init=/sbin/overlay-init
  │     overlay_root=vdb
  │                                          Firecracker boots kernel
  │                                            │
  │                                          overlay-init (PID 1, shell)
  │                                            mount squashfs (ro base)
  │                                            mount ext4 (rw, config only)
  │                                            mount -t overlay (stacked)
  │                                            pivot_root → merged
  │                                            exec /sbin/init
  │                                            │
  │                                          systemd (PID 1)
  │                                            │
  │                                            ├─ motlie-vmm-guest.service
  │                                            │    ExecStart=motlie-vmm-bootstrap
  │                                            │      │
  ├─ vsock listener (port 5000)  ◄─────────────│      ├─ connect vsock:5000
  │   HandshakeMsg::BinaryRequest              │      │   handshake: BinaryRequest
  │   → send guest agent binary (~3MB)         │      │   receive binary → /tmp (tmpfs)
  │                                            │      │   exec into guest agent (same PID)
  │                                            │      │
  ├─ FsServer ready  ◄────────────────────────│      ├─ read /etc/motlie-vmm/mounts.json
  │   (mounts registered, overlay layers set)  │      │
  │                                            │      ├─ for each mount in config:
  │   HandshakeMsg::Fs { tag }  ◄──────────────│      │   connect vsock:5000
  │   → VsockConnectionHandler::serve()        │      │   handshake: Fs { tag }
  │                                            │      │   VsockFuseMount::new(stream, tag)
  │                                            │      │   fuser::mount2(mount, path, opts)
  │                                            │      │
  │   HandshakeMsg::Control  ◄─────────────────│      ├─ connect vsock:5000
  │   → control loop                           │      │   handshake: Control
  │                                            │      │   send ControlMsg::Ready
  │                                            │      │   listen for AddMount/RemoveMount
  │                                            │      │
  │                                            │      └─ FUSE mounts active
  │                                            │
  │                                            └─ sshd.service (CA-based auth)
  │
  └─ VM ready, bridge SSH session
```

### Guest Agent Code with motlie-vfs

The guest agent (~700 lines in the VMM doc) uses motlie-vfs's vsock composite on the
client side. With the library, the FUSE mount code becomes a thin wrapper:

```rust
// motlie-vmm-guest main.rs (inside the VM)
use motlie_vfs::vsock::VsockFuseMount;

const HOST_CID: u32 = 2;
const VMM_PORT: u32 = 5000;

fn main() {
    let config: MountConfig = read_json("/etc/motlie-vmm/mounts.json");

    // Spawn one FUSE mount per configured path
    let mut handles = Vec::new();
    for mount in &config.mounts {
        let tag = mount.tag.clone();
        let guest_path = mount.guest_path.clone();
        let read_only = mount.read_only;

        let handle = std::thread::Builder::new()
            .name(format!("fuse-{}", tag))
            .spawn(move || {
                // 1. Connect to host vsock
                let mut stream = vsock_connect(HOST_CID, VMM_PORT).unwrap();

                // 2. VMM-level handshake (not part of motlie-vfs)
                send_handshake(&mut stream, &HandshakeMsg::Fs { tag: tag.clone() });

                // 3. Hand the stream to motlie-vfs for FUSE mounting
                std::fs::create_dir_all(&guest_path).ok();
                let fuse_mount = VsockFuseMount::new(stream, &tag);
                let mut opts = vec![
                    fuser::MountOption::AutoUnmount,
                    fuser::MountOption::AllowRoot,
                ];
                if read_only {
                    opts.push(fuser::MountOption::RO);
                }
                fuser::mount2(fuse_mount, &guest_path, &opts).unwrap();
            })
            .unwrap();
        handles.push(handle);
    }

    // Control connection (stays in motlie-vmm, not in the library)
    let mut control = vsock_connect(HOST_CID, VMM_PORT).unwrap();
    send_handshake(&mut control, &HandshakeMsg::Control);
    send_msg(&mut control, &ControlMsg::Ready);

    // Listen for dynamic mount commands
    loop {
        match recv_msg(&mut control) {
            Ok(ControlMsg::AddMount { tag, guest_path, read_only }) => {
                // Same pattern: connect vsock, handshake, VsockFuseMount
                let handle = spawn_fuse_mount(&tag, &guest_path, read_only);
                handles.push(handle);
                send_msg(&mut control, &ControlMsg::MountReady { tag });
            }
            Ok(ControlMsg::RemoveMount { tag }) => {
                let path = find_mount_path(&tag);
                Command::new("fusermount").args(["-u", &path]).status().ok();
            }
            Ok(ControlMsg::Shutdown) => break,
            Err(_) => break,
            _ => {}
        }
    }
}
```

The boundary is clear:
- **motlie-vfs** owns: `VsockFuseMount` (FUSE ↔ vsock bridge), frame encoding, `fuser` integration
- **motlie-vmm guest** owns: vsock connect, `HandshakeMsg` handshake, control loop, mount orchestration
- **motlie-vmm host** owns: vsock listener, `HandshakeMsg` dispatch, `FsServer` + overlay setup, control protocol

### Host-Side Setup: FsServer + Overlay for a VM

On the host side, the VMM daemon creates the `FsServer` with mounts and overlay content
before the guest agent connects. This is where the in-memory overlay is used to inject
content that should never exist on disk.

```rust
// Inside motlie-vmm daemon, during ensure_vm("alice"):

use motlie_vfs::core::FsServer;
use motlie_vfs::vsock::VsockConnectionHandler;

// Build FsServer for this user's VM
let server = FsServer::builder()
    .mount("home", home_dir.join("alice"), false)            // /root → ~/alice
    .mount("scratch", scratch_dir.join("alice"), false)      // /tmp
    .overlay(true)
    .events(4096)
    .policy(credential_policy)
    .build()?;

// Inject SSH keys into the "home" mount's overlay.
// On disk, ~/alice/.ssh/ may not exist or may be empty.
// The overlay makes these files appear at /root/.ssh/ inside the VM.
// If ~/alice/.ssh/ doesn't exist on disk, the server creates a synthetic
// directory inode for /.ssh/ implicitly from the child entries below.
let overlay = server.overlay().unwrap();
overlay.put_layer("credentials", 0)?;
overlay.put("credentials", "home", "/.ssh/authorized_keys", alice_pubkey)?;
overlay.put("credentials", "home", "/.ssh/config", ssh_config_bytes)?;
overlay.put("credentials", "home", "/.ssh/id_ed25519", alice_private_key)?;
overlay.put("credentials", "home", "/.ssh/id_ed25519.pub", alice_public_key)?;

// API keys as a synthetic .env file
overlay.put("credentials", "home", "/.env", Bytes::from(format!(
    "ANTHROPIC_API_KEY={}\nOPENAI_API_KEY={}",
    anthropic_key, openai_key,
)))?;

// Write mounts.json for guest agent to read from overlay ext4
let mounts_config = MountConfig {
    mounts: vec![
        MountEntry { tag: "home", guest_path: "/root", read_only: false },
        MountEntry { tag: "scratch", guest_path: "/tmp", read_only: false },
    ],
};
inject_into_overlay(&overlay_ext4, "/etc/motlie-vmm/mounts.json", &mounts_config)?;

// Start vsock listener — dispatch connections to FsServer
let server = Arc::new(server);
tokio::spawn({
    let server = server.clone();
    async move {
        loop {
            let stream = vsock_accept(cid, 5000).await?;
            let handshake: HandshakeMsg = recv_handshake(&mut stream).await?;
            match handshake {
                HandshakeMsg::Fs { tag } => {
                    let handler = VsockConnectionHandler::new(&server, &tag);
                    tokio::spawn(async move { handler.serve(stream).await });
                }
                HandshakeMsg::BinaryRequest => { /* send guest agent binary */ }
                HandshakeMsg::Control => { /* control loop */ }
            }
        }
    }
});
```

### Example: SSH Keys in Overlay, Home Directory on Disk

This example shows the motivating use case for the overlay: the guest mounts the user's
entire home directory from the host via FUSE pass-through, but the `.ssh` subdirectory
is populated from an in-memory overlay. The SSH private keys never exist on the host
filesystem — they're injected by the VMM daemon from a secure source (vault, env var,
per-session generation).

**What the guest sees at /root:**

```
/root/                          ← FUSE mount, tag "home" → ~/alice on host
├── projects/                   ← pass-through to ~/alice/projects (disk)
├── .config/                    ← pass-through to ~/alice/.config (disk)
├── .bashrc                     ← pass-through to ~/alice/.bashrc (disk)
├── .ssh/                       ← IN-MEMORY OVERLAY (credentials layer)
│   ├── authorized_keys         ← overlay: never on disk
│   ├── config                  ← overlay: never on disk
│   ├── id_ed25519              ← overlay: never on disk
│   ├── id_ed25519.pub          ← overlay: never on disk
│   └── known_hosts             ← overlay: initially empty, writes captured in-memory
├── .env                        ← IN-MEMORY OVERLAY (synthetic file, not on disk)
└── documents/                  ← pass-through to ~/alice/documents (disk)
```

**How handle_op() resolves these paths:**

```
read("/root/projects/README.md")
  → overlay.resolve("home", "/projects/README.md") → None
  → std::fs::read("~/alice/projects/README.md")     ← disk

read("/root/.ssh/id_ed25519")
  → overlay.resolve("home", "/.ssh/id_ed25519") → Some(("credentials", key_bytes))
  → return key_bytes                              ← in-memory, never touches disk

read("/root/.env")
  → overlay.resolve("home", "/.env") → Some(("credentials", env_bytes))
  → return env_bytes                  ← synthetic file, ~/alice/.env doesn't exist

write("/root/.ssh/known_hosts", new_entry)
  → overlay resolves "/.ssh/known_hosts" → exists in "credentials" layer
  → write updates the in-memory layer    ← disk untouched

write("/root/projects/src/main.rs", code)
  → overlay.resolve("home", "/projects/src/main.rs") → None
  → std::fs::write("~/alice/projects/src/main.rs")   ← goes to disk

ls("/root/")
  → readdir on host: ~/alice/ → [projects, .config, .bashrc, documents, ...]
  → overlay entries for "home" at depth 1: [.ssh, .env]
  → merge per-name: .ssh exists on both? overlay entry wins for that name
  →                  .env only in overlay? synthetic entry added
  → result: [projects, .config, .bashrc, documents, .ssh, .env, ...]

ls("/root/.ssh/")
  → readdir on host: ~/alice/.ssh/ → [known_hosts, old_config]
  → overlay entries at /.ssh/*: [id_ed25519, id_ed25519.pub, config, authorized_keys]
  → merge per-name: config exists on both? overlay wins
  →                  known_hosts only on disk? passes through
  →                  id_ed25519 only in overlay? synthetic entry added
  → result: [known_hosts, old_config, id_ed25519, id_ed25519.pub, config, authorized_keys]
```

**readdir merging with overlay and policy filtering:**

When `handle_op()` processes a `Readdir`, it merges entries and then filters:

1. Read real entries from `std::fs::read_dir(host_path)`.
   (If directory is synthetic — doesn't exist on disk — skip this step.)
2. Collect overlay entries whose path is a direct child of the queried directory.
3. Merge by name: overlay entry shadows disk entry with the same name.
4. Disk entries not shadowed pass through unchanged.
5. Overlay entries with no disk counterpart appear as synthetic entries.
6. **Policy filter:** each merged entry is passed through `policy.check(Readdir, tag, path)`.
   Entries denied by policy are excluded from the result. This enables directory-level
   lockdown without modifying the overlay.

### Overlay Scope: File-Level Granularity

The overlay operates at **file granularity**. Each `put()` targets a specific file path.
`readdir` merges disk and overlay entries per-file:

```
readdir("/.ssh/")
  disk entries from ~/alice/.ssh/:  [known_hosts, old_config]
  overlay entries at /.ssh/*:       [id_ed25519, id_ed25519.pub, config, authorized_keys]

  merge rules:
    - overlay entry shadows disk entry with the same name (config replaces old_config? no,
      "config" shadows disk "config" if both exist; "old_config" passes through)
    - disk entries not shadowed by overlay pass through unchanged
    - overlay entries with no disk counterpart appear as synthetic files

  result: [known_hosts, old_config, id_ed25519, id_ed25519.pub, config, authorized_keys]
```

This allows mixed directories: some files from disk, some from overlay, in the same
directory listing. An overlaid file shadows only that file; siblings are unaffected.

**Security via policy, not overlay:**

If a caller wants full control over a directory (e.g., no unexpected disk files in `.ssh/`),
that's a policy concern, not an overlay concern. Use `PolicyFn` to block access to
non-overlaid files within sensitive directories. Because `policy.check()` is called for
each `Readdir` entry (step 6 above), denied entries are excluded from directory listings
as well as from direct access:

```rust
struct SshLockdown { overlay: Arc<MemOverlay> }

impl PolicyFn for SshLockdown {
    fn check(&self, op: FsOp, tag: &str, path: &str) -> Result<(), i32> {
        // In /.ssh/: only serve files that exist in the overlay.
        // This filters both direct access (Lookup, Read, Open) AND
        // Readdir entries — disk files in .ssh/ are invisible.
        if path.starts_with("/.ssh/") && self.overlay.resolve(tag, path).is_none() {
            return Err(libc::ENOENT);
        }
        Ok(())
    }
}
```

With this policy, `readdir("/.ssh/")` returns only overlay entries — disk files like
`known_hosts` or `old_config` are filtered out. Without this policy, they pass through.
The overlay always merges; the policy decides what's visible.

This separates concerns cleanly: overlay handles content injection and merging; policy
handles access control. They compose independently.

### Credential Lifecycle

```
VM create:
  1. VMM daemon generates or fetches credentials (SSH keys, API tokens)
  2. overlay.put("credentials", "home", "/.ssh/id_ed25519", key) — in memory only
  3. Guest agent boots, mounts /root via FUSE
  4. SSH inside the VM reads /root/.ssh/id_ed25519 → overlay serves it from memory
  5. Credentials never touch host disk or guest disk

VM stop:
  6. FsServer dropped → overlay dropped → credentials gone from memory
  7. Host dir ~/alice/ unchanged (no .ssh/ written)
  8. No cleanup needed — nothing was persisted

VM restart:
  9. VMM daemon re-creates FsServer, re-injects credentials into overlay
  10. Guest sees the same /root/.ssh/ as before
```

This is a significant improvement over the original VMM design, which used separate FUSE
mounts for each credential directory (`.claude/`, `.config/gh/`, `.codex/`, `.npmrc`).
With the overlay, a single "home" mount serves the entire home directory, and credentials
are injected as overlay entries within it. Fewer mounts, fewer vsock connections, simpler
guest agent, and the credentials never touch any filesystem.

### Comparison: Original VMM Design vs. Overlay Model

| Aspect | Original (separate mounts) | Overlay model |
|--------|---------------------------|---------------|
| Mount count per VM | 6+ (workspace, scratch, 4 cred dirs) | 2 (home, scratch) |
| vsock connections per VM | 6+ (one per mount) | 2 (one per mount) |
| Credential storage | Host disk (`~/.motlie-vmm/creds/`) | In-memory overlay (never on disk) |
| Credential persistence | Survives VM restart (on-disk) | Must be re-injected on restart |
| Guest agent complexity | One thread per mount | Same, but fewer mounts |
| Adding a new cred dir | New mount tag, new vsock connection | `overlay.put()` call |
| Readdir at /root | Only shows disk files | Merges disk + overlay entries |

The overlay model trades credential persistence (which the original had via on-disk cred
dirs) for stronger security (credentials never on disk). For OAuth tokens that refresh
automatically, this is fine — the VMM daemon re-injects them from a secure store on each
VM boot. For API keys set via environment variables, the overlay injects them as a
synthetic `.env` file.

## Alternatives Considered

### Alternative A: 9P2000.L Protocol

Use the Plan 9 filesystem protocol (Linux variant) instead of a custom wire format.

**Pros:**
- Mature, well-specified protocol with existing implementations
- Linux kernel has a native 9P client (`mount -t 9p`) -- no FUSE needed on Linux
- QEMU/virtio-9p uses it for host directory sharing

**Cons:**
- No native macOS kernel client -- would still need FUSE on macOS, defeating the "one
  protocol" goal
- 9P's walk/attach/clunk semantics are more complex than needed for our tag-based model
- No natural event emission point -- 9P servers are typically transparent passthrough
- Existing Rust 9P libraries (jmbaur/p9, oxidecomputer/p9) are incomplete or dormant
- Protocol overhead: 9P's per-message type headers and walk chains add unnecessary bytes
  for our use case where the client already knows the mount root

**Verdict:** Rejected. The native Linux kernel client is appealing but the macOS gap
means we'd need two code paths. Our workload (known mount points, event interception)
is better served by a thinner custom protocol.

### Alternative B: FUSE Passthrough (virtio-fs model)

Forward raw FUSE protocol messages over the transport, as virtiofsd does with virtio-fs.

**Pros:**
- Zero translation overhead -- FUSE messages pass through unmodified
- Proven approach in production (virtiofsd, kata containers)
- Potentially simpler client (just forward /dev/fuse messages)

**Cons:**
- FUSE protocol is Linux-specific -- message format tied to kernel version and platform.
  macOS FUSE-T uses a different internal protocol (NFSv4). Would need platform-specific
  server-side decoding, which defeats the purpose.
- FUSE protocol messages are not self-describing without kernel version context
- Event emission requires parsing FUSE messages on the server side anyway
- `fuser` already handles the /dev/fuse ↔ Rust translation -- using it on the client side
  and speaking a clean protocol over the wire is simpler than raw FUSE passthrough
- No support for pluggable encoding -- locked to FUSE wire format

**Verdict:** Rejected. The platform asymmetry (Linux FUSE vs macOS FUSE-T internals) makes
raw passthrough impractical for cross-platform use. The `fuser` crate already absorbs the
platform differences; our custom protocol sits cleanly above it.

### Alternative C: Custom Protocol (Selected)

A purpose-built, minimal FS protocol with length-prefixed frames, pluggable encoding, and
tag-based routing. The client translates FUSE ops to protocol frames; the server translates
frames to host FS ops.

**Pros:**
- Thinnest possible protocol for our use case
- Natural event emission point (server decodes every frame)
- Tag-based routing built into the handshake
- Pluggable encoding (bincode now, msgpack or protobuf later)
- Platform-agnostic (no Linux or macOS kernel assumptions in the wire format)
- Transport-agnostic (any AsyncRead+AsyncWrite)
- Composable: vsock composite skips Frame overhead entirely for the VM path

**Cons:**
- Custom protocol means custom bugs -- no battle-tested implementations to lean on
- Must implement every FS operation ourselves (9P and FUSE passthrough get some for free)
- No interop with non-Rust clients without reimplementing the protocol

**Verdict:** Selected. The cons are manageable: the protocol is small (~15 operations),
the server is straightforward (translate to `std::fs` calls), and non-Rust interop is not
a current requirement. The event emission and policy enforcement capabilities, which are
core to the motlie-vmm use case, fit naturally into this model. The composable architecture
means the VM path (vsock) pays no overhead for the RPC protocol layer it doesn't use.

## Components and Testing

### Components to Test

| Component | What to test | Method |
|-----------|-------------|--------|
| `core::op` | FsOp/FsResult serde round-trip | Unit tests |
| `core::server` | handle_op() for all FsOp variants against real tempdir | Integration tests with `tempfile` |
| `core::inode` | Inode allocation, lookup, refcount, eviction | Unit tests |
| `core::event` | Event emission on FS ops, no-subscriber zero-cost | Unit + integration |
| `core::policy` | Deny returns correct errno, allow passes through | Unit tests |
| `core::overlay` | Layer priority, put/get/remove, resolve order, synthetic files in readdir | Unit tests |
| `core::overlay` | Write capture to overlaid path updates layer, not disk | Integration tests with `tempfile` |
| `control::socket` | LAYER/PUT/RM/GET/LS commands over Unix socket | Integration tests |
| `control::protocol` | Line protocol parse/format round-trip | Unit tests |
| `vsock::handler` | VsockConnectionHandler serve loop over duplex | Integration tests |
| `vsock::mount` | VsockFuseMount translates FUSE ops correctly | Unit tests (mock stream) |
| `rpc::frame` | Frame/FrameBody serde round-trip | Unit tests |
| `rpc::codec` | Codec encode/decode for all variants | Unit tests |
| `rpc::io` | Length-prefixed read/write, max frame size enforcement | Unit tests with `tokio::io::duplex` |
| `rpc::server` | RpcServer handshake + request/response cycle | Integration tests |
| `rpc::client` | RpcClient handshake + request/response cycle | Integration tests |
| End-to-end (vsock) | VsockConnectionHandler + VsockFuseMount over duplex, real FUSE mount | Integration (requires FUSE) |
| End-to-end (rpc) | RpcServer + RpcClient over Unix socket, real FUSE mount | Integration (requires FUSE) |
| Cross-platform | macOS FUSE-T mount via RPC composite | Manual test (documented procedure) |

### Test Utilities

```rust
/// Direct testing: no transport, no serde.
#[cfg(test)]
let server = FsServer::builder().mount("t", dir, false).build()?;
let result = server.handle_op("t", FsOp::Lookup { parent: 1, name: "foo".into() });

/// RPC testing: in-memory duplex, no FUSE.
#[cfg(test)]
pub fn rpc_test_pair(server: FsServer, tag: &str) -> (JoinHandle<()>, RpcClient) {
    let (cs, ss) = tokio::io::duplex(64 * 1024);
    let rpc = RpcServer::new(server, BincodeCodec);
    let handle = tokio::spawn(async move { rpc.serve(ss).await.unwrap() });
    let client = RpcClient::connect(cs, BincodeCodec, tag).await.unwrap();
    (handle, client)
}

/// vsock testing: same pattern, duplex stream.
#[cfg(test)]
pub fn vsock_test_pair(server: FsServer, tag: &str) -> (JoinHandle<()>, VsockFuseMount) {
    let (cs, ss) = tokio::io::duplex(64 * 1024);
    let handler = VsockConnectionHandler::new(&server, tag);
    let handle = tokio::spawn(async move { handler.serve(ss).await.unwrap() });
    let mount = VsockFuseMount::new(cs, tag);
    (handle, mount)
}
```

## Dependencies

### Required (always compiled -- server-core)

| Crate | Purpose |
|-------|---------|
| `serde`, `serde_derive` | FsOp/FsResult serialization |
| `bytes` | Zero-copy byte buffers in data ops |
| `tokio` (features: io-util, sync, rt) | Async I/O, broadcast channel |
| `thiserror` | Error types |
| `tracing` | Structured logging |

### Feature-Gated

| Crate | Feature flag | Purpose |
|-------|-------------|---------|
| `bincode` | `bincode-codec` | Default wire encoding (used by vsock and default rpc) |
| `fuser` | `client` | FUSE filesystem implementation |
| `tokio-vsock` | `vsock` | vsock stream type |
| `tokio-rustls`, `rustls` | `tls` | TLS for TCP transport |

### Dev Dependencies

| Crate | Purpose |
|-------|---------|
| `tempfile` | Temporary directories for server FS tests |
| `tokio` (features: test-util, macros) | Test runtime |
