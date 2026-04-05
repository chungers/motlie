# motlie-vfs Policy Engine: Enriched Observability and Access Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-05 | @claude-tl | PolicyEvaluator plumbing: builder→FsServer→dispatch, chaining examples with step-by-step walkthroughs (normal read, scanning, staging) |
| 2026-04-05 | @claude-tl | Drop Observe, two-action PolicyAction (Allow/Deny + confidence), deny_threshold, emission truth table, appendix updated |
| 2026-04-05 | @claude-tl | Confidence on PolicyAction, FsInterceptPolicy with ArcSwap, PolicyEvent at call site, remove EventSinkPolicy |
| 2026-04-05 | @claude-tl | Stateless policy redesign: dispatch-layer state, caller identity (UID/GID/PID) with FUSE plumbing gap, pre-computed signals, fd hijacking/execve limitations, Setattr mode/ownership tracking |
| 2026-04-05 | @claude-tl | Add 3-phase detection framework (discovery→staging→execution), enriched FsOpContext (inode, is_sensitive, parent_path), write_entropy in FsOpResult, detector examples |
| 2026-04-05 | @claude-tl | Initial DESIGN: FsPolicy trait, PolicyAction with errno, chaining, enriched events, OTel logging, event sink policy |

## Problem Statement

The current vfs policy layer (`PolicyFn` trait, `AllowAll` default) provides
a minimal pre-operation gate that returns `Ok(())` or `Err(errno)`. The event
system (`FsEvent`) emits operation notifications but leaves path, bytes, and
result fields unpopulated.

This is sufficient for v1 but insufficient for production observability and
access control:

1. **No decision context**: policy sees `(op, tag, path)` but not whether
   the path is overlay-managed or what layer the operation targets.
2. **No post-operation visibility**: no way to observe results, measure
   latency, or track bytes actually transferred.
3. **No composition**: can't chain policies (e.g. logging + access control).
4. **No structured reasons**: denied operations return raw `errno` with no
   audit trail of *why*.
5. **Dynamic dispatch overhead**: `Box<dyn PolicyFn>` on every operation
   even when the policy is trivial.

### Goals

1. **Enriched context**: pre-op callbacks see operation kind, tag, path,
   overlay status. Post-op callbacks see result, bytes, latency.
2. **Chainable policies**: `.chain()` method on the trait. First Deny wins,
   min-confidence propagated, Allow defers. Same chaining pattern used by
   `motlie-vnet`'s `EgressPolicy`.
3. **Zero-cost default**: `NoPolicy` compiles out entirely. Generic
   `P: FsPolicy` threaded through `FsServer<P>`.
4. **Structured audit**: `PolicyAction::Deny { errno, reason }` instead
   of raw errno. PolicyReason logged with structured tracing fields.
5. **Logging as a policy**: `LoggingPolicy` emits OTel-compatible tracing
   spans. Not special-cased — just a chainable policy that returns Allow.
6. **Populated events**: `FsEvent` gains result status, actual bytes, path,
   and latency — populated by the dispatch layer, not the policy.
7. **Event sink policy**: a generic event producer policy that sends
   `FsOpContext` to a caller-supplied closure, enabling cross-stack
   correlation in the host runtime.

### Non-Goals

- Content-level inspection of file data (read/write payload)
- Encryption or integrity checking of filesystem data
- User-facing policy configuration language (host runtime's concern)
- Cross-stack policy correlation (host runtime composes; see "Future" section)

## Migration from PolicyFn

The existing `PolicyFn` trait is replaced by `FsPolicy`:

| Before (`PolicyFn`) | After (`FsPolicy`) |
|---------------------|---------------------|
| `fn check(&self, op, tag, path) -> Result<(), i32>` | `fn on_op(&self, ctx: &FsOpContext) -> PolicyAction` |
| `Ok(())` | `PolicyAction::Allow { reason: None, confidence: 1.0 }` |
| `Err(libc::EROFS)` | `PolicyAction::Deny { errno: libc::EROFS, reason: ... }` |
| `Box<dyn PolicyFn>` | Generic `P: FsPolicy` |
| No post-op | `fn on_complete(&self, ctx, result)` |
| No chaining | `.chain(next)` |
| `AllowAll` | `NoPolicy` (zero-cost, compiled out) |

## Policy API

### PolicyAction

```rust
/// Policy decision. Binary: Allow or Deny. Confidence carries the
/// uncertainty signal — no separate "Observe" variant.
/// Shared shape with motlie-vnet's EgressPolicy.
pub enum PolicyAction {
    /// Allow the operation to proceed.
    Allow {
        /// Why this was allowed. The chain keeps the reason from the
        /// lowest-confidence policy (the weakest link).
        reason: Option<PolicyReason>,
        /// Confidence (0.0–1.0). Below deny_threshold → auto-converted
        /// to Deny at the dispatch layer.
        confidence: f64,
    },
    /// Deny the operation. `errno` is returned to the guest via FUSE.
    Deny {
        errno: i32,
        reason: PolicyReason,
        confidence: f64,
    },
}
```

**Confidence propagation:** the chain propagates the minimum confidence
across all policies — the composite's confidence is its weakest link.
Same semantics as vnet. See DESIGN-VNET-POLICY for the full confidence
model (rule-based, heuristic, ML, session-aware use cases).

### PolicyReason

```rust
/// Reason attached to a policy decision. Zero-alloc for categories.
pub enum PolicyReason {
    /// Filesystem-specific category.
    Category(FsCategory),
    /// Custom reason.
    Custom(Cow<'static, str>),
}

pub enum FsCategory {
    ReadOnlyMount,     // mount is read-only
    CredentialAccess,  // .ssh/, .env, secrets
    OverlayManaged,    // path is overlay-injected
    SystemPath,        // /etc, /proc, /sys
    UserData,          // user workspace files
}
```

### Predefined (errno, reason) Pairs

| errno | FsCategory | Use case |
|-------|-----------|----------|
| `EROFS` | `ReadOnlyMount` | Write to read-only mount |
| `EACCES` | `CredentialAccess` | Non-overlay access to credential path |
| `EPERM` | `SystemPath` | Blocked system path mutation |
| `EACCES` | `OverlayManaged` | Unauthorized overlay bypass |

### FsPolicy Trait

```rust
/// Filesystem policy callbacks. All methods have default allow-all impls.
///
/// Policies are **stateless evaluators** — they receive fully-enriched
/// context (with dispatch-layer pre-computed signals) and return a
/// decision. Cross-operation state is maintained by the dispatch layer,
/// not by policies.
///
/// `Send + Sync` required because FsServer may be accessed from multiple
/// vsock connection handler threads concurrently.
pub trait FsPolicy: Send + Sync + 'static {
    /// Called before every filesystem operation. Return Deny to prevent
    /// the operation from executing (guest sees the errno).
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    /// Called after every filesystem operation completes. The operation
    /// has already executed — this is observe-only. Used for logging,
    /// metrics, latency tracking, and audit trails.
    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    /// Chain another policy after this one.
    ///
    /// Evaluation order: `self` first, then `next`.
    /// - First Deny wins (short-circuit) for `on_op`.
    /// - Minimum confidence propagated (weakest link).
    /// - Allow defers to next.
    /// - `on_complete` does NOT short-circuit — both policies always
    ///   see completions (the operation already executed).
    fn chain<N: FsPolicy>(self, next: N) -> impl FsPolicy
    where
        Self: Sized;
}
```

**Chaining usage:**

```rust
let policy = LoggingPolicy
    .chain(InterceptPolicy::new(blocked.clone()))
    .chain(ScanDetector)
    .chain(CategoryGuard);

// `policy` implements FsPolicy. The developer never names or sees
// the intermediate types — the compiler resolves them.
```

Chain evaluation rules:
- First `Deny` wins (short-circuit)
- Minimum confidence propagated across the chain (weakest link)
- `Allow` defers to next
- Minimum confidence propagated across the chain
- `on_complete` does NOT short-circuit — all policies see completions

### Stateless Policy Design

Policy callbacks are **stateless evaluators** — they receive fully-enriched
context and return a decision. All cross-operation state (rate counters,
inode history, path tracking) is maintained by the **dispatch layer**,
not by individual policies. This mirrors vnet's interceptor/policy
separation: the interceptor owns the DNS reverse map and flow tracker;
the policy just sees enriched context.

Benefits:
- Policies are pure functions of context — no `RwLock`, no `HashMap`,
  no interior mutability
- The same state enrichment benefits every policy in the chain
- State lifetime is managed by the dispatch layer (not leaked through
  policy `Drop` impls)
- Cross-stack correlation happens in the external event sink, not inside
  per-stack policies

### Context Types

```rust
/// Context provided to on_op() before the operation executes.
/// All stateful signals are pre-computed by the dispatch layer.
#[derive(Clone)]
pub struct FsOpContext {
    /// Operation kind.
    pub op: FsOpKind,
    /// Mount tag (e.g. "alice-home").
    pub tag: String,
    /// Inode number. Enables cross-operation correlation via the dispatch
    /// layer's inode tracker (e.g. read-then-write-same-inode patterns).
    pub inode: u64,
    /// Mount-relative path (e.g. "/.ssh/authorized_keys").
    pub path: String,
    /// Parent directory path (for Create, Mkdir, Unlink, Rename).
    pub parent_path: Option<String>,
    /// Whether this path is overlay-managed (injected by the host).
    pub is_overlay: bool,
    /// Whether this path matches a predefined sensitive pattern
    /// (.ssh/, .env, credentials, secrets). Pre-classified by dispatch.
    pub is_sensitive: bool,
    /// Timestamp when the operation was received.
    pub timestamp: Instant,

    // --- Caller identity (from FUSE request header) ---

    /// Guest-side caller UID. Extracted from the kernel FUSE request
    /// header (fuse_in_header.uid). Available for ALL operations once
    /// the guest client plumbing gap is closed (see below).
    pub caller_uid: Option<u32>,
    /// Guest-side caller GID.
    pub caller_gid: Option<u32>,
    /// Guest-side caller PID. Enables per-process policy decisions:
    /// "apt (PID 1234) reading /etc/apt/ = Allow" vs
    /// "unknown (PID 9999) reading /.ssh/ = Allow { confidence: 0.3 }".
    pub caller_pid: Option<u32>,

    // --- Dispatch-layer pre-computed signals ---

    /// Metadata operation rate for this tag (ops/sec for Lookup/Getattr/
    /// Readdir in the current window). Computed by the dispatch layer's
    /// rate tracker. Policies just compare against thresholds.
    pub metadata_rate: u32,
    /// Unique paths accessed in the current time window.
    pub unique_paths_in_window: u32,
    /// For Write ops: whether this inode was recently Read (read-then-write
    /// pattern, computed by dispatch inode tracker).
    pub preceded_by_read: bool,
    /// Bytes read from this inode in the preceding read (if any).
    pub preceding_read_bytes: Option<usize>,
    /// For Setattr ops: mode change (old_mode, new_mode). Detects
    /// permission escalation (e.g. chmod +x on a newly-written file).
    pub mode_change: Option<(u32, u32)>,
    /// For Setattr ops: ownership change (old_uid, old_gid, new_uid, new_gid).
    pub ownership_change: Option<(u32, u32, u32, u32)>,
}

/// Result provided to on_complete() after the operation executes.
#[derive(Clone)]
pub struct FsOpResult {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Errno if the operation failed (0 on success).
    pub errno: i32,
    /// Bytes transferred (for Read/Write operations).
    pub bytes: Option<usize>,
    /// Shannon entropy of written data (Write operations only).
    /// Computed from the write buffer during dispatch — the server
    /// already handles this data, so entropy is metadata, not DPI.
    pub write_entropy: Option<f64>,
    /// Wall-clock latency of the operation.
    pub latency: Duration,
}
```

`FsOpContext` and `FsOpResult` are `Clone + Send` to support the event
sink policy.

### Dispatch-Layer State (not policy state)

The dispatch layer maintains cross-operation state and injects derived
signals into `FsOpContext` before calling the policy chain:

```
FsServer dispatch layer:
├── RateTracker      → metadata_rate, unique_paths_in_window
├── InodeTracker     → preceded_by_read, preceding_read_bytes
├── SensitiveClassifier → is_sensitive
├── ModeTracker      → mode_change (for Setattr)
└── OwnershipTracker → ownership_change (for Setattr)
```

This state is opaque to policies. Policies see only the pre-computed
fields in `FsOpContext`. No `RwLock` or `HashMap` in policy impls.

### Design Rationale

**Why `inode` matters:** Filesystem attacks correlate operations on the
same file. A process that reads a file then immediately writes high-entropy
data back to the same inode is likely encrypting it in place (ransomware).
The dispatch layer's inode tracker computes `preceded_by_read` — the
policy just checks a bool.

**Why `write_entropy` is not DPI:** Unlike network payload inspection
(which requires TLS termination), filesystem write data passes through
the vfs server in plaintext — the server already copies it from the FUSE
buffer to disk. Computing entropy during dispatch is ~50ns and provides
a strong signal without inspecting content semantics.

**Why `is_sensitive` is pre-classified:** Every policy in the chain would
otherwise recompute the same path-matching logic. Pre-classifying once
in the dispatch layer avoids redundant work.

**Why `caller_pid` is `Option`:** The FUSE protocol provides PID per
request, but the guest client doesn't forward it yet. See "FUSE Process
Identity Gap" below.

## FUSE Process Identity Gap

### What the Kernel Provides

The Linux FUSE driver includes caller identity in every request header
(`fuse_in_header`):
- **UID** (user ID) — always available
- **GID** (group ID) — always available
- **PID** (process ID) — always available

The `fuser` crate (v0.15.1, used by the guest FUSE client) exposes
all three per request — confirmed in source:
- `Request::uid() -> u32` (`fuser-0.15.1/src/request.rs:658`)
- `Request::gid() -> u32` (`fuser-0.15.1/src/request.rs:664`)
- `Request::pid() -> u32` (`fuser-0.15.1/src/request.rs:670`)

These read directly from the kernel's `fuse_in_header`. There is no
protocol limitation — the data is available, just not forwarded.

### What motlie-vfs Currently Forwards

| Operation | UID/GID | PID | Status |
|-----------|---------|-----|--------|
| `Access` | Yes | No | Forwarded, used for permission check |
| `Create` | Yes | No | Forwarded, used for ownership |
| `Mkdir` | Yes | No | Forwarded, used for ownership |
| `Getlk`/`Setlk` | No | Yes (from param) | PID from lock param, not from `Request` |
| All other 20 ops | No | No | `_req` parameter ignored |

### Implementation Gap (4 layers, each self-contained)

**Layer 1: Guest FUSE client** (`libs/vfs/src/client/fuse.rs`)

The `FuseClient` implements the `fuser::Filesystem` trait. Each method
receives a `Request` parameter. Currently 21 of 24 methods prefix it
with `_req` (intentionally ignored). Change: extract identity from all.

Operations that need `_req` → `req` + extraction:
```
lookup (line 72), getattr (line 88), setattr (line 111),
setxattr (line 149), getxattr (line 172), listxattr (line 192),
removexattr (line 201), open (line 243), read (line 255),
write (line 278), readdir (line 284), unlink (line 430),
rmdir (line 439), readlink (line 476), rename (line 484),
symlink (line 498), release (line 517), fsync (line 525),
statfs (line 537), getlk (line 344), setlk (line 371)
```

Each becomes: `let caller = CallerIdentity::from_req(&req);`

**Layer 2: FsOp wire type** (`libs/vfs/src/core/op.rs`)

Preferred: wrapper struct to avoid modifying every FsOp variant:
```rust
#[derive(Serialize, Deserialize)]
pub struct CallerIdentity {
    pub uid: u32,
    pub gid: u32,
    pub pid: u32,
}

#[derive(Serialize, Deserialize)]
pub struct IdentifiedOp {
    pub caller: CallerIdentity,
    pub op: FsOp,
}
```

The existing `uid`/`gid` fields in `Access`, `Create`, `Mkdir` become
redundant — the caller identity is always in the wrapper. These fields
can be deprecated but kept for backward compatibility during migration.

**Layer 3: Wire format** (`libs/vfs/src/vsock/client.rs`)

Currently serializes `FsOp` directly (line 34):
```rust
let encoded = bincode::serde::encode_to_vec(op, bincode::config::standard())
```

Change to serialize `IdentifiedOp`. For backward compatibility, use a
version byte or message-type tag so the host server can accept both
old (`FsOp`) and new (`IdentifiedOp`) messages during migration.

**Layer 4: Host server dispatch** (`libs/vfs/src/core/server.rs`)

`handle_op()` (line 155) currently receives `(tag, op: FsOp)`.
Change to receive `(tag, identified_op: IdentifiedOp)`, extract
`CallerIdentity`, and populate `FsOpContext.caller_uid/gid/pid`.

For old clients that send bare `FsOp`, use `CallerIdentity::default()`
with uid=0, gid=0, pid=0 (root, unknown process) — the `Option<u32>`
fields in `FsOpContext` become `None` when identity is not available.

### Effort Estimate

| Layer | Files changed | Lines (est.) | Risk |
|-------|--------------|-------------|------|
| Guest FUSE client | 1 | ~50 (mechanical: `_req` → `req`, add extraction) | Low |
| FsOp wire type | 1 | ~15 (new struct + derive) | Low |
| Wire format | 1 | ~10 (serialize wrapper instead of bare FsOp) | Medium (backward compat) |
| Host dispatch | 1 | ~20 (extract identity, populate context) | Low |
| **Total** | **4** | **~95** | **Low–Medium** |

The mechanical nature of the change (21 identical `_req` → `req`
substitutions) makes it low-risk. The wire format change is the only
medium-risk step due to backward compatibility.

### What PID Enables

With PID available, the policy API supports:

```rust
// Per-process policy decisions (stateless — dispatch pre-computes)
fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
    // Known package manager PID accessing package cache = Allow
    // Unknown PID accessing /.ssh/ = Allow { confidence: 0.3 }
    if ctx.is_sensitive && !is_known_process(ctx.caller_pid) {
        PolicyAction::Allow { reason: ... }
    } else {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

Note: `is_known_process()` requires a process allowlist, which is a
host-runtime concern (not vfs). The vfs API provides the PID; the
host runtime interprets it.

### What Remains Non-Detectable

Even with full PID/UID/GID:

| Scenario | Why non-detectable |
|----------|-------------------|
| **fd hijacking** (process B reads from A's fd) | Kernel FUSE sees the original opener (A), not the hijacker (B). The fd table is guest-kernel state invisible to the FUSE protocol. |
| **Binary execution** (`execve`) | Not a filesystem operation through the FUSE mount. Kernel syscall, not FUSE. |
| **Pipe to shell** (`curl \| sh`) | Process I/O, not filesystem I/O. stdin pipe doesn't touch the FUSE mount. |
| **Process ancestry** | FUSE provides PID but not parent PID or process tree. Would need `/proc` inspection in the guest agent. |

**Scope-limited visibility:**

| Scenario | Why limited | Path to coverage |
|----------|------------|-----------------|
| **Unix domain socket creation** (`bind()` on `/tmp/evil.sock`) | Only visible if the socket path is under a FUSE-mounted tag. In motlie's architecture, `/tmp`, `/var/run`, `/dev/shm` are on the guest's root filesystem (squashfs + ext4 overlay), not FUSE-mounted. | Could be covered if mount scope expands to include `/tmp` — but that changes motlie-vfs from "targeted host directory mounts" to "full guest filesystem proxy," which is a different architecture. |
| **Network socket creation** (`socket(AF_INET)` + `connect()`) | Kernel network stack, not filesystem. | This is vnet's domain — `on_tcp_connect()` sees it. Cross-stack correlation (via PolicyEvent channels) bridges the gap. |

These are guest-kernel and mount-scope limitations, not vfs API
limitations. They match vnet's documented limitations (encrypted payload
invisible, guest process name requires vsock agent).

### Roadmap Context

The policy capabilities described in this DESIGN build on the current
v1.2 motlie-vfs implementation. Future versions expand the testing and
enforcement surface:

- **v1.3** (`libs/vmm/examples`): SSH proxy and automated harness,
  enabling automated scenario testing of both vnet and vfs policies
  in composed guest VM flows.
- **v1.4** (TBD): Extended mount scope, policy prototyping for paths
  currently outside FUSE visibility (`/tmp`, `/var/run`). May require
  guest-side cooperation (mount delegation or guest agent extensions).

### NoPolicy (zero-cost default)

```rust
pub struct NoPolicy;

impl FsPolicy for NoPolicy {
    // All default impls → Allow { reason: None }
}
```

When `P = NoPolicy`, the compiler eliminates all policy code in dispatch.

### LoggingPolicy (OTel-compatible tracing)

Emits spans compatible with OpenTelemetry exporters. If the host runtime
wires `tracing-opentelemetry`, events automatically flow to Jaeger/OTLP.

```rust
pub struct LoggingPolicy;

impl FsPolicy for LoggingPolicy {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "INTERNAL",
            otel.name = %format!("vfs.{:?}", ctx.op),
            vfs.op = ?ctx.op,
            vfs.tag = %ctx.tag,
            vfs.path = %ctx.path,
            vfs.overlay = ctx.is_overlay,
            "fs.op.start"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        tracing::info!(
            otel.kind = "INTERNAL",
            otel.name = %format!("vfs.{:?}", ctx.op),
            otel.status_code = if result.success { "OK" } else { "ERROR" },
            vfs.op = ?ctx.op,
            vfs.tag = %ctx.tag,
            vfs.path = %ctx.path,
            vfs.bytes = ?result.bytes,
            vfs.latency_us = result.latency.as_micros() as u64,
            vfs.errno = result.errno,
            "fs.op.end"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

### CredentialGuard (reference implementation)

```rust
pub struct CredentialGuard;

impl FsPolicy for CredentialGuard {
    fn on_op(&self, ctx: &FsOpContext) -> FsAction {
        if is_credential_path(&ctx.path) && !ctx.is_overlay {
            PolicyAction::Deny {
                errno: libc::EACCES,
                reason: PolicyReason::Category(FsCategory::CredentialAccess),
            }
        } else {
            PolicyAction::Allow { reason: None, confidence: 1.0 }
        }
    }
}

fn is_credential_path(path: &str) -> bool {
    path.starts_with("/.ssh/") || path.starts_with("/.env")
        || path.contains("/credentials") || path.contains("/secrets")
}
```

### InterceptPolicy (externally-fed blocklist)

A policy whose deny set is managed by an external correlator thread.
The policy reads from a concurrent `ArcSwap` snapshot (~25ns, zero
reader-writer contention). The correlator writes atomically from a
separate thread.

```rust
use arc_swap::ArcSwap;

pub struct FsInterceptPolicy {
    /// Blocked paths. Updated atomically by the correlator.
    blocked: Arc<ArcSwap<HashSet<String>>>,
}

impl FsPolicy for FsInterceptPolicy {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        let snapshot = self.blocked.load();
        if snapshot.contains(&ctx.path) {
            PolicyAction::Deny {
                errno: libc::EACCES,
                reason: PolicyReason::Custom("correlator: blocked path".into()),
                confidence: 1.0,
            }
        } else {
            PolicyAction::Allow { reason: None, confidence: 1.0 }
        }
    }
}
```

**Correlator writes (separate thread):**

```rust
// Clone-on-write: amortized O(1) insert, O(1) atomic swap
let mut current = (**blocklist.load()).clone();
current.insert("/tmp/suspicious_payload".to_string());
blocklist.store(Arc::new(current));
```

**Latency analysis — why ArcSwap:**

| Approach | Hot-path read | Correlator write | Contention |
|----------|-------------|-----------------|-----------|
| `Arc<RwLock<HashSet>>` | ~50ns | ~100ns | Writer blocks readers |
| `try_recv` + local map | ~50ns | ~20ns | None, but 1-call delay |
| **`ArcSwap<HashSet>`** | **~25ns** | ~50ns + O(N) clone | **Zero contention** |

The O(N) clone happens on the correlator thread (off hot path) and only
when the blocklist changes — rare compared to per-operation frequency.

**Feedback loop:**

```
Dispatch thread (hot path)       Correlator thread (slow path)
    │                                    │
    ├─ policy.on_op(&ctx)               │
    ├─ emit PolicyEvent ──────────────→ │ receives event
    ├─ enforce action                    │ accumulates state
    │                                    │ computes confidence
    │                                    │ if confidence > threshold:
    │   ┌────────────────────────────── │   blocklist.store(new_set)
    │   │  ArcSwap (~25ns read)         │
    │   ▼                                │
    ├─ InterceptPolicy reads snapshot   │
    ├─ Deny if blocked                  │
```

## Anomaly Detection by Filesystem Phase

Filesystem attacks follow a timeline analogous to network connection
phases. Detection moves from **discovery** (what is the attacker looking
for?), to **staging** (how is data being prepared?), to **execution**
(how does data leave or get destroyed?).

All detection is implemented via stateful `FsPolicy` implementations
using `on_op()` and `on_complete()` — no additional callbacks needed.
The enriched `FsOpContext` (inode, is_sensitive, parent_path) and
`FsOpResult` (write_entropy, bytes) provide the fields each detector
needs.

### Phase 1: Discovery (Scanning & Enumeration)

Before stealing data, an attacker must find it. Characterized by
"breadth over depth" — many metadata operations across sensitive paths.

**Policy callbacks:** `on_op()` for pre-op gate, `on_complete()` for
rate tracking.

| Technique | Method | Detection Pattern | API fields |
|-----------|--------|-------------------|-----------|
| **Directory Crawling** | Rapid Lookup/Getattr/Readdir across sensitive paths | High-frequency metadata access: single session querying thousands of files in seconds | `ctx.op` (Lookup/Getattr/Readdir), `ctx.timestamp` → stateful rate counter per tag |
| **Configuration Hunting** | Targeting `.env`, `.ssh/id_rsa`, `config.toml` | Any non-overlay access to known-secret file paths | `ctx.is_sensitive`, `ctx.is_overlay` → deny if `is_sensitive && !is_overlay` |
| **Breadth Scanning** | Accessing many distinct paths in a short window | Unique path count per time window exceeds threshold | `ctx.path`, `ctx.timestamp` → stateful `HashSet<String>` per window |

**Example: ScanDetector (stateless — dispatch pre-computes rates)**

```rust
struct ScanDetector;

impl FsPolicy for ScanDetector {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        // Dispatch layer pre-computes metadata_rate and unique_paths_in_window.
        // No RwLock, no HashMap — just compare against thresholds.
        if ctx.metadata_rate > 500 || ctx.unique_paths_in_window > 200 {
            PolicyAction::Allow {
                reason: PolicyReason::Custom(
                    format!("scan: {} ops/s, {} unique paths",
                            ctx.metadata_rate, ctx.unique_paths_in_window).into()
                ),
            }
        } else {
            PolicyAction::Allow { reason: None, confidence: 1.0 }
        }
    }
}
```

**API coverage:** Fully supported. Dispatch layer owns the rate tracker
and path set — policy is a pure function of pre-computed context fields.

### Phase 2: Staging (Modification & Collection)

Data is prepared for theft — moved, compressed, or encrypted to a
staging location.

**Policy callbacks:** `on_op()` for path deviation, `on_complete()` for
write analysis.

| Technique | Method | Detection Pattern | API fields |
|-----------|--------|-------------------|-----------|
| **Archive Creation** | `tar`, `zip`, `7z` bundling directories into one file | Sudden spike in write bytes producing a single high-entropy file | `result.bytes`, `result.write_entropy` → large write + entropy > 4.0 |
| **Data Siphoning** | Copying to `/tmp/.hidden/` or `/var/tmp/` | Files written to world-writable directories by processes that usually stay in `/home` | `ctx.path` (starts with `/tmp/`, `/var/tmp/`), `ctx.op` (Create/Write) |
| **MIME-Type Mismatch** | Renaming `.zip` to `.txt` to bypass DLP | File extension doesn't match the entropy/magic of the content | `ctx.path` (extension), `result.write_entropy` (high entropy + `.txt` extension) |

**Example: StagingDetector**

```rust
struct StagingDetector;

impl FsPolicy for StagingDetector {
    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        if ctx.op != FsOpKind::Write {
            return PolicyAction::Allow { reason: None, confidence: 1.0 };
        }
        // Large high-entropy write to temp directory = staging
        if let (Some(bytes), Some(entropy)) = (result.bytes, result.write_entropy) {
            let is_temp = ctx.path.starts_with("/tmp/") || ctx.path.starts_with("/var/tmp/");
            if is_temp && bytes > 1_000_000 && entropy > 4.0 {
                return PolicyAction::Allow {
                    reason: PolicyReason::Custom(
                        format!("staging pattern: {}MB high-entropy write to {}",
                                bytes / 1_000_000, ctx.path).into()
                    ),
                };
            }
        }
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

**API coverage:** Fully supported. `FsOpResult.write_entropy` is the
key enabler — it detects compressed/encrypted staging without inspecting
content semantics. `FsOpResult.bytes` and `FsOpContext.path` detect
path deviation and volume anomalies.

### Phase 3: Execution/Exit (Exfiltration or Destruction)

The "final act" — data leaves the system or is rendered useless.

**Policy callbacks:** `on_op()` + `on_complete()` with cross-operation
correlation via `ctx.inode`.

| Technique | Method | Detection Pattern | API fields |
|-----------|--------|-------------------|-----------|
| **In-Place Encryption** | Read file, encrypt, write back (ransomware) | Read-write symmetry: process reads a block then writes high-entropy block to same inode | `ctx.inode` → stateful: track (inode, last_read_bytes), then `on_complete` for Write to same inode with high `write_entropy` |
| **File Shredding** | Overwriting with null bytes or random data | Entropy collapse: write_entropy ≈ 0.0 (all zeros) or ≈ 8.0 (random) followed by Unlink | `result.write_entropy` (near 0 or near max), then `ctx.op` = Unlink on same inode |
| **Bulk Delete** | Rapid deletion of many files | High-rate Unlink/Rmdir operations in a short window | `ctx.op` (Unlink/Rmdir), `ctx.timestamp` → stateful rate counter |

**Example: RansomwareDetector (stateless — dispatch pre-computes inode history)**

```rust
struct RansomwareDetector;

impl FsPolicy for RansomwareDetector {
    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        if ctx.op != FsOpKind::Write {
            return PolicyAction::Allow { reason: None, confidence: 1.0 };
        }
        // Dispatch layer pre-computes: was this inode recently Read?
        // How many bytes? Policy just checks the pre-computed fields.
        if let (Some(write_bytes), Some(entropy)) = (result.bytes, result.write_entropy) {
            if ctx.preceded_by_read {
                if let Some(read_bytes) = ctx.preceding_read_bytes {
                    let similar_size = (read_bytes as f64 - write_bytes as f64).abs()
                        / (read_bytes as f64).max(1.0) < 0.2;
                    if similar_size && entropy > 4.0 {
                        return PolicyAction::Deny {
                            errno: libc::EACCES,
                            reason: PolicyReason::Custom(
                                format!("ransomware: read {}B → write {}B \
                                         (entropy {:.1}) inode {}",
                                        read_bytes, write_bytes, entropy,
                                        ctx.inode).into()
                            ),
                        };
                    }
                }
            }
        }
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

**API coverage:** Fully supported. `preceded_by_read` and
`preceding_read_bytes` are pre-computed by the dispatch layer's inode
tracker. `write_entropy` detects encryption. The policy is a pure
function — no state, no locks. No additional callbacks needed.

### Phase Summary

```
Guest fs operations timeline:

  Lookup Lookup Getattr Readdir    Create Write Write    Read Write Unlink
  ─────────────────────────────    ──────────────────    ──────────────────
       Phase 1                       Phase 2                Phase 3
       Discovery                     Staging                Execution

  ┌─────────────────┐        ┌─────────────────┐      ┌─────────────────┐
  │ ScanDetector    │        │ StagingDetector  │      │ RansomDetector  │
  │                 │        │                  │      │                 │
  │ Rate counting   │        │ Write entropy    │      │ Inode tracking  │
  │ Path breadth    │        │ Path deviation   │      │ Read-write sym  │
  │ Sensitive flag  │        │ Volume anomaly   │      │ Entropy shift   │
  │                 │        │                  │      │                 │
  │ on_op()         │        │ on_complete()    │      │ on_complete()   │
  │ on_complete()   │        │                  │      │ + stateful map  │
  └─────────────────┘        └─────────────────┘      └─────────────────┘
```

### API Coverage Summary

| Detector | Phase | Callback | Key fields | Coverage |
|----------|-------|----------|-----------|----------|
| Directory crawling | Discovery | `on_op` | `op`, `timestamp` | Full |
| Configuration hunting | Discovery | `on_op` | `is_sensitive`, `is_overlay` | Full |
| Breadth scanning | Discovery | `on_op` | `path`, `timestamp` | Full |
| Archive creation | Staging | `on_complete` | `bytes`, `write_entropy` | Full |
| Data siphoning | Staging | `on_op` | `path` (temp dirs) | Full |
| MIME mismatch | Staging | `on_complete` | `path` (ext), `write_entropy` | Full |
| In-place encryption | Execution | `on_complete` | `inode`, `write_entropy`, `bytes` | Full |
| File shredding | Execution | `on_complete` | `write_entropy` (near 0/max), then Unlink | Full |
| Bulk delete | Execution | `on_op` | `op` (Unlink/Rmdir), `timestamp` | Full |

All nine detection techniques are implementable with the two existing
callbacks (`on_op`, `on_complete`) and the enriched context types. No
additional callbacks needed.

## Policy Event Emission

Event emission is **infrastructure at the call site** (`FsServer::dispatch`),
not a policy concern. Policies are pure decision functions.

### PolicyEvent Enum

```rust
/// Emitted by FsServer::dispatch after every policy evaluation.
pub enum PolicyEvent {
    FsOp {
        ctx: FsOpContext,
        action: PolicyAction,
    },
    FsComplete {
        ctx: FsOpContext,
        result: FsOpResult,
        action: PolicyAction,
    },
}
```

### Emission at the Call Site

```rust
// In FsServer::dispatch — NOT inside a policy:
let action = self.policy.on_op(&ctx);           // pure decision

if let Some(ref tx) = self.event_tx {            // infrastructure
    let _ = tx.try_send(PolicyEvent::FsOp {
        ctx: ctx.clone(),
        action: action.clone(),
    });
}

match action {                                    // enforcement
    PolicyAction::Deny { errno, .. } => FsResult::Error { errno },
    _ => {
        let result = self.execute_op(mount, op);
        let completion_action = self.policy.on_complete(&ctx, &op_result);
        if let Some(ref tx) = self.event_tx {
            let _ = tx.try_send(PolicyEvent::FsComplete {
                ctx, result: op_result, action: completion_action,
            });
        }
        result
    }
}
```

### Cross-Stack Correlation

The host runtime subscribes to both vfs and vnet PolicyEvent channels:

```rust
enum RuntimeEvent {
    Net(motlie_vnet::PolicyEvent),
    Fs(motlie_vfs::PolicyEvent),
}
// Correlator thread reads from unified channel, sees both stacks.
```

Neither crate depends on the other. The host runtime owns the unified
enum and the channel.

## PolicyEvaluator and Plumbing

### PolicyEvaluator

The runtime bundle that sits between `FsServer::dispatch()` and the
policy chain. Owns the chain, the threshold, and the event channel.
The handler calls `evaluate()` and `complete()` — threshold and emission
are internal.

```rust
pub struct PolicyEvaluator<P: FsPolicy> {
    policy: P,
    deny_threshold: f64,
    event_tx: Option<mpsc::Sender<PolicyEvent>>,
}

impl<P: FsPolicy> PolicyEvaluator<P> {
    /// Pre-op: evaluate chain, apply threshold, emit event.
    pub fn evaluate(&self, ctx: &FsOpContext) -> PolicyAction {
        let action = self.policy.on_op(ctx);
        let action = self.apply_threshold(action);
        self.emit(PolicyEvent::FsOp { ctx: ctx.clone(), action: action.clone() }, &action);
        action
    }

    /// Post-op: all policies see the completion.
    pub fn complete(&self, ctx: &FsOpContext, result: &FsOpResult) {
        let action = self.policy.on_complete(ctx, result);
        self.emit(PolicyEvent::FsComplete {
            ctx: ctx.clone(), result: result.clone(), action: action.clone(),
        }, &action);
    }

    fn apply_threshold(&self, action: PolicyAction) -> PolicyAction {
        match action {
            PolicyAction::Allow { confidence, reason } if confidence < self.deny_threshold => {
                PolicyAction::Deny {
                    errno: libc::EACCES,
                    reason: reason.unwrap_or(PolicyReason::Custom("below threshold".into())),
                    confidence,
                }
            }
            other => other,
        }
    }

    fn emit(&self, event: PolicyEvent, action: &PolicyAction) {
        if let Some(ref tx) = self.event_tx {
            let dominated = match action {
                PolicyAction::Deny { .. } => true,
                PolicyAction::Allow { confidence, .. } => *confidence < 1.0,
            };
            if dominated {
                let _ = tx.try_send(event);
            }
        }
    }
}
```

### Builder → FsServer

```rust
pub struct FsServerBuilder<P: FsPolicy = NoPolicy> {
    policy: P,
    deny_threshold: f64,
    event_tx: Option<mpsc::Sender<PolicyEvent>>,
    // ... other FsServer config ...
}

impl FsServerBuilder<NoPolicy> {
    pub fn new() -> Self {
        Self {
            policy: NoPolicy,
            deny_threshold: 0.0,
            event_tx: None,
        }
    }
}

impl<P: FsPolicy> FsServerBuilder<P> {
    /// Set the policy chain. Changes the generic parameter.
    pub fn policy<Q: FsPolicy>(self, policy: Q) -> FsServerBuilder<Q> {
        FsServerBuilder {
            policy,
            deny_threshold: self.deny_threshold,
            event_tx: self.event_tx,
        }
    }

    pub fn deny_threshold(mut self, threshold: f64) -> Self {
        self.deny_threshold = threshold;
        self
    }

    pub fn policy_events(mut self, tx: mpsc::Sender<PolicyEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    pub fn build(self) -> FsServer<P> {
        let policy_evaluator = if /* P is NoPolicy and no threshold and no events */ {
            None
        } else {
            Some(PolicyEvaluator {
                policy: self.policy,
                deny_threshold: self.deny_threshold,
                event_tx: self.event_tx,
            })
        };

        FsServer {
            policy_evaluator,
            // ... mounts, overlay, etc ...
        }
    }
}
```

### FsServer dispatch integration

```rust
pub struct FsServer<P: FsPolicy = NoPolicy> {
    policy_evaluator: Option<PolicyEvaluator<P>>,
    // ... mounts, overlay, etc ...
}

impl<P: FsPolicy> FsServer<P> {
    fn dispatch(&self, mount: &MountState, op: &FsOp) -> FsResult {
        if mount.read_only && is_write_op(op) {
            return FsResult::Error { errno: libc::EROFS };
        }

        if let Some(ref evaluator) = self.policy_evaluator {
            let ctx = self.build_context(mount, op);
            let action = evaluator.evaluate(&ctx);

            match action {
                PolicyAction::Deny { errno, .. } => FsResult::Error { errno },
                PolicyAction::Allow { .. } => {
                    let start = Instant::now();
                    let result = self.execute_op(mount, op);
                    let op_result = self.build_result(&result, start);
                    evaluator.complete(&ctx, &op_result);
                    result
                }
            }
        } else {
            // No evaluator — zero overhead (single null-check, ~1ns)
            self.execute_op(mount, op)
        }
    }
}
```

### deny_threshold

```rust
let server = FsServer::builder()
    .policy(my_chain)
    .deny_threshold(0.5)   // Allow with confidence < 0.5 → Deny
    .build();
```

Default: `0.0` (only explicit Denys block). Strict mode: `0.8`.

## Chaining Example: LoggingPolicy + ScanDetector

### Code

```rust
// --- LoggingPolicy: always Allow, logs via OTel tracing ---
pub struct LoggingPolicy;

impl FsPolicy for LoggingPolicy {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "INTERNAL",
            otel.name = %format!("vfs.{:?}", ctx.op),
            vfs.op = ?ctx.op,
            vfs.tag = %ctx.tag,
            vfs.path = %ctx.path,
            vfs.overlay = ctx.is_overlay,
            vfs.sensitive = ctx.is_sensitive,
            "fs.op"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        tracing::info!(
            otel.kind = "INTERNAL",
            otel.name = %format!("vfs.{:?}", ctx.op),
            otel.status_code = if result.success { "OK" } else { "ERROR" },
            vfs.op = ?ctx.op,
            vfs.tag = %ctx.tag,
            vfs.path = %ctx.path,
            vfs.bytes = ?result.bytes,
            vfs.latency_us = result.latency.as_micros() as u64,
            vfs.write_entropy = ?result.write_entropy,
            "fs.op.complete"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}

// --- ScanDetector: flags high-rate metadata scanning ---
pub struct ScanDetector;

impl FsPolicy for ScanDetector {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        // Dispatch layer pre-computed metadata_rate and unique_paths_in_window
        if ctx.metadata_rate > 500 || ctx.unique_paths_in_window > 200 {
            PolicyAction::Allow {
                reason: Some(PolicyReason::Custom(
                    format!("scan: {} ops/s, {} paths", ctx.metadata_rate,
                            ctx.unique_paths_in_window).into()
                )),
                confidence: 0.3,  // suspicious but not blocking
            }
        } else {
            PolicyAction::Allow { reason: None, confidence: 1.0 }
        }
    }

    fn on_complete(&self, _ctx: &FsOpContext, _result: &FsOpResult) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}

// --- Chain and configure ---
let server = FsServer::builder()
    .policy(
        LoggingPolicy                    // logs every operation
            .chain(ScanDetector)         // flags scanning patterns
    )
    .deny_threshold(0.5)
    .policy_events(event_tx)
    .build();
```

**Note:** `LoggingPolicy` goes first because it always returns
`Allow { confidence: 1.0 }`. If `ScanDetector` were first and a future
enforcement policy returned `Deny`, `LoggingPolicy` would never run.
Putting the logger first ensures every operation is traced.

### Walkthrough: normal file read (known-good)

```
1. LoggingPolicy.on_op(ctx { op: Read, path: "/src/main.rs", metadata_rate: 5 })
   → tracing::info!(vfs.op = Read, vfs.path = "/src/main.rs", ...)
   → returns Allow { confidence: 1.0 }

2. ScanDetector.on_op(ctx)
   → metadata_rate 5 < 500 and unique_paths 3 < 200
   → returns Allow { confidence: 1.0 }

3. Chain merges: both Allow { 1.0 }, min = 1.0
   → composite: Allow { reason: None, confidence: 1.0 }

4. PolicyEvaluator.apply_threshold: Allow 1.0 ≥ any threshold → pass through

5. PolicyEvaluator.emit: confidence 1.0 → NOT emitted (not interesting)

6. FsServer: execute_op → read file → return data to guest
```

### Walkthrough: rapid directory scanning (suspicious)

```
1. LoggingPolicy.on_op(ctx { op: Lookup, path: "/.ssh/id_rsa", metadata_rate: 800 })
   → tracing::info!(vfs.op = Lookup, vfs.path = "/.ssh/id_rsa", ...)
   → returns Allow { confidence: 1.0 }

2. ScanDetector.on_op(ctx)
   → metadata_rate 800 > 500
   → returns Allow { reason: Custom("scan: 800 ops/s, 150 paths"), confidence: 0.3 }

3. Chain merges: Allow 1.0 vs Allow 0.3, min = 0.3
   → composite: Allow { reason: Custom("scan: 800 ops/s, 150 paths"), confidence: 0.3 }
   → reason from ScanDetector (weakest link)

4. PolicyEvaluator.apply_threshold(0.5):
   → confidence 0.3 < threshold 0.5? → YES
   → converted to: Deny { errno: EACCES, reason: Custom("scan: 800 ops/s..."), confidence: 0.3 }

5. PolicyEvaluator.emit: Deny → always emitted
   → PolicyEvent::FsOp { ctx, action: Deny { scan pattern, 0.3 } }
   → correlator receives: "rapid scanning pattern detected, 800 ops/s"

6. FsServer: return FsResult::Error { errno: EACCES } to guest
   Guest sees: Lookup of /.ssh/id_rsa → Permission denied
```

### Walkthrough: write to temp directory (staging detection via on_complete)

```
1. LoggingPolicy.on_op(ctx { op: Write, path: "/tmp/payload.bin" })
   → tracing::info!(...)
   → returns Allow { confidence: 1.0 }

2. ScanDetector.on_op(ctx)
   → metadata_rate 10 < 500 → normal
   → returns Allow { confidence: 1.0 }

3. Chain merges: Allow { 1.0 }

4. Threshold: passes. Emit: not interesting (confidence 1.0).

5. FsServer: execute_op → write succeeds

6. PolicyEvaluator.complete(ctx, result { write_entropy: Some(4.8), bytes: Some(5_000_000) })
   
   LoggingPolicy.on_complete(ctx, result)
   → tracing::info!(vfs.write_entropy = 4.8, vfs.bytes = 5000000, ...)
   → returns Allow { confidence: 1.0 }

   ScanDetector.on_complete(ctx, result)
   → returns Allow { confidence: 1.0 }

   (A StagingDetector in the chain would see write_entropy 4.8 + /tmp/ path
   and return Allow { confidence: 0.2 } — but ScanDetector doesn't check writes)

7. Emit: post-op Allow 1.0 → not emitted.
   (If StagingDetector were chained, confidence 0.2 → would be emitted)
```

## Enriched FsEvent

The existing `FsEvent` struct is enriched with currently-empty fields:

```rust
pub struct FsEvent {
    pub timestamp: SystemTime,
    pub tag: String,
    pub op_kind: FsOpKind,
    pub path: String,            // NOW POPULATED: mount-relative path
    pub bytes: Option<usize>,    // NOW POPULATED: for Read/Write ops
    pub success: bool,           // NEW: operation result
    pub errno: i32,              // NEW: 0 on success
    pub latency_us: u64,         // NEW: wall-clock microseconds
}
```

Populated by the dispatch layer regardless of which policy is active.
Events and policy are independent — events always flow, policy decides
what happens.

## Future: Cross-Stack Event Correlation

The `PolicyEvent` emission in both vfs and vnet enables a host-runtime
correlator that sees both filesystem and network activity. This is NOT
a vfs concern — the host runtime owns composition:

- vfs emits: `PolicyEvent::FsOp` / `PolicyEvent::FsComplete`
- vnet emits: `PolicyEvent::DnsQuery` / `PolicyEvent::TcpConnect` / etc.
- Host runtime wraps both in a `RuntimeEvent` enum
- Correlator reads the unified channel, writes to `InterceptPolicy`
  blocklists (ArcSwap) in both stacks

Example correlation signals (host runtime, not vfs):
- "read `/.env` then DNS lookup for `pastebin.com`" → exfiltration
- "write to `/tmp/payload` then connect to unknown IP:4444" → reverse shell

The vfs DESIGN requirement is: **context types must be `Clone + Send`**
so the call-site emission can clone them into the event channel.

## Detection Primitives

Network detection primitives (entropy analysis, domain parsing, rate
tracking) live in `motlie_vnet::policy` — they are network-specific.

Filesystem-specific detection utilities (e.g. path pattern matching for
credential paths) live in `motlie_vfs::policy` when implemented. Each
crate owns its own detection primitives. If a truly stack-agnostic
primitive emerges in the future, it can be extracted to a shared crate
at that point — but not speculatively.

## Components and Testing

| Component | Test approach |
|-----------|--------------|
| `FsPolicy` trait + `NoPolicy` | Unit: returns Allow for all ops |
| Chain evaluation (on_op) | Unit: Deny short-circuits, min-confidence propagation, reason from weakest link |
| Chain evaluation (on_complete) | Unit: both policies always see completions |
| `LoggingPolicy` | Unit: returns Allow, tracing events match OTel conventions |
| `CredentialGuard` | Unit: denies non-overlay credential paths, allows overlay |
| `InterceptPolicy` | Unit: reads ArcSwap blocklist, denies blocked paths |
| `PolicyEvent` emission | Integration: events emitted at dispatch call site |
| `FsOpContext` population | Integration: dispatch populates path, overlay status |
| `FsOpResult` population | Integration: dispatch measures latency, captures bytes |
| `FsEvent` enrichment | Integration: events carry path, bytes, result, latency |
| Zero-cost NoPolicy | Build: binary size delta ≈ 0 vs current code |

---

## Appendix: How Chain Evaluation Works Internally

*This section is technical background for implementers. Developers using
the policy API never interact with these internals — they only call
`.chain()` on the trait and get back an `impl FsPolicy`.*

When a developer writes:

```rust
let policy = LoggingPolicy.chain(ScanDetector).chain(CategoryGuard);
```

The compiler constructs a nested type. Conceptually:

```
policy: Chain<LoggingPolicy, Chain<ScanDetector, CategoryGuard>>
```

The actual struct name is internal to the crate and not exported. Each
"link" in the chain holds two fields: `current` (this policy) and `next`
(the rest of the chain). The struct implements `FsPolicy` by calling
`self.current` first, then `self.next`:

```rust
// Internal to the crate — not public API:
struct ChainedFsPolicy<A, B> { current: A, next: B }

impl<A: FsPolicy, B: FsPolicy> FsPolicy for ChainedFsPolicy<A, B> {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        let first = self.current.on_op(ctx);
        match first {
            PolicyAction::Deny { .. } => first,  // short-circuit
            PolicyAction::Allow { confidence: c1, reason: r1 } => {
                let second = self.next.on_op(ctx);
                match second {
                    PolicyAction::Deny { .. } => second,
                    PolicyAction::Allow { confidence: c2, reason: r2 } => {
                        // Min confidence, reason from weakest link
                        if c2 < c1 {
                            PolicyAction::Allow { confidence: c2, reason: r2 }
                        } else {
                            PolicyAction::Allow { confidence: c1, reason: r1 }
                        }
                    }
                }
            }
        }
    }

    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        // No short-circuit: both policies always see completions.
        let _ = self.current.on_complete(ctx, result);
        self.next.on_complete(ctx, result)
    }
}
```

Because the type is fully generic (no `Box<dyn>`), the compiler
monomorphizes the entire chain at compile time. Each `.on_op()` call
inlines through the nesting — there is no vtable dispatch, no heap
allocation, and no dynamic method resolution on the hot path.

When `P = NoPolicy`, the compiler sees that all methods return
`PolicyAction::Allow { confidence: 1.0 }` and eliminates the entire
chain evaluation as dead code. The dispatch loop compiles to the same
machine code as if policy support did not exist.
| Migration compat | Integration: existing tests pass with NoPolicy default |
