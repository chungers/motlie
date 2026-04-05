# motlie-vfs Policy Engine: Enriched Observability and Access Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
   Observe is sticky, Allow defers. Same chaining pattern used by
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
| `Ok(())` | `PolicyAction::Allow { reason: None }` |
| `Err(libc::EROFS)` | `PolicyAction::Deny { errno: libc::EROFS, reason: ... }` |
| `Box<dyn PolicyFn>` | Generic `P: FsPolicy` |
| No post-op | `fn on_complete(&self, ctx, result)` |
| No chaining | `.chain(next)` |
| `AllowAll` | `NoPolicy` (zero-cost, compiled out) |

## Policy API

### PolicyAction

```rust
/// Policy decision. Shared shape with motlie-vnet's EgressPolicy.
pub enum PolicyAction {
    /// Allow the operation.
    Allow { reason: Option<PolicyReason> },
    /// Deny the operation. `errno` is returned to the guest via FUSE.
    Deny { errno: i32, reason: PolicyReason },
    /// Allow but flag for observation.
    Observe { reason: PolicyReason },
}
```

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
        PolicyAction::Allow { reason: None }
    }

    /// Called after every filesystem operation completes. The operation
    /// has already executed — this is observe-only. Used for logging,
    /// metrics, latency tracking, and audit trails.
    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Chain another policy after this one.
    ///
    /// Evaluation order: `self` first, then `next`.
    /// - First Deny wins (short-circuit) for `on_op`.
    /// - Observe is sticky — propagates even if next returns Allow.
    /// - Allow defers to next.
    /// - `on_complete` does NOT short-circuit — both policies always
    ///   see completions (the operation already executed).
    fn chain<N: FsPolicy>(self, next: N) -> impl FsPolicy
    where
        Self: Sized,
    {
        ChainedFsPolicy { current: self, next }
    }
}
```

The internal `ChainedFsPolicy<A, B>` struct is not part of the public API.
Users interact with it only through `impl FsPolicy`:

```rust
let policy = LoggingPolicy.chain(CredentialGuard);
// type: impl FsPolicy (opaque)
```

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
    /// "unknown (PID 9999) reading /.ssh/ = Observe".
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

The `fuser` crate (v0.15, used by the guest FUSE client) exposes
`Request::uid()`, `Request::gid()`, and `Request::pid()`.

### What motlie-vfs Currently Forwards

| Operation | UID/GID | PID | Status |
|-----------|---------|-----|--------|
| `Access` | Yes | No | Forwarded, used for permission check |
| `Create` | Yes | No | Forwarded, used for ownership |
| `Mkdir` | Yes | No | Forwarded, used for ownership |
| `Getlk`/`Setlk` | No | Yes (from param) | PID from lock param, not from `Request` |
| All other 20 ops | No | No | `_req` parameter ignored |

### Implementation Gap

To plumb full caller identity through the stack:

1. **Guest FUSE client** (`libs/vfs/src/client/fuse.rs`):
   - For all 24 operations, extract `req.uid()`, `req.gid()`, `req.pid()`
   - Currently only 3 operations extract uid/gid; 21 prefix `req` with `_`

2. **FsOp enum** (`libs/vfs/src/core/op.rs`):
   - Option A: Add `uid: u32, gid: u32, pid: u32` to every variant
   - Option B (preferred): Add a wrapper struct:
     ```rust
     pub struct CallerIdentity {
         pub uid: u32,
         pub gid: u32,
         pub pid: u32,
     }

     pub struct IdentifiedOp {
         pub caller: CallerIdentity,
         pub op: FsOp,
     }
     ```
     This avoids modifying every FsOp variant.

3. **Wire format** (`libs/vfs/src/vsock/`):
   - Serialize `IdentifiedOp` instead of `FsOp` over vsock
   - Wire-compatible change if using a new message type (existing clients
     send `FsOp`, new clients send `IdentifiedOp`)

4. **Host server dispatch** (`libs/vfs/src/core/server.rs`):
   - Extract `CallerIdentity` from the deserialized message
   - Pass to `FsOpContext.caller_uid`, `caller_gid`, `caller_pid`

### What PID Enables

With PID available, the policy API supports:

```rust
// Per-process policy decisions (stateless — dispatch pre-computes)
fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
    // Known package manager PID accessing package cache = Allow
    // Unknown PID accessing /.ssh/ = Observe
    if ctx.is_sensitive && !is_known_process(ctx.caller_pid) {
        PolicyAction::Observe { reason: ... }
    } else {
        PolicyAction::Allow { reason: None }
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

These are guest-kernel limitations, not vfs design limitations. They
match vnet's documented limitations (encrypted payload invisible,
guest process name requires vsock agent).

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
        PolicyAction::Allow { reason: None }
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
        PolicyAction::Allow { reason: None }
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
            PolicyAction::Allow { reason: None }
        }
    }
}

fn is_credential_path(path: &str) -> bool {
    path.starts_with("/.ssh/") || path.starts_with("/.env")
        || path.contains("/credentials") || path.contains("/secrets")
}
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
            PolicyAction::Observe {
                reason: PolicyReason::Custom(
                    format!("scan: {} ops/s, {} unique paths",
                            ctx.metadata_rate, ctx.unique_paths_in_window).into()
                ),
            }
        } else {
            PolicyAction::Allow { reason: None }
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
            return PolicyAction::Allow { reason: None };
        }
        // Large high-entropy write to temp directory = staging
        if let (Some(bytes), Some(entropy)) = (result.bytes, result.write_entropy) {
            let is_temp = ctx.path.starts_with("/tmp/") || ctx.path.starts_with("/var/tmp/");
            if is_temp && bytes > 1_000_000 && entropy > 4.0 {
                return PolicyAction::Observe {
                    reason: PolicyReason::Custom(
                        format!("staging pattern: {}MB high-entropy write to {}",
                                bytes / 1_000_000, ctx.path).into()
                    ),
                };
            }
        }
        PolicyAction::Allow { reason: None }
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
            return PolicyAction::Allow { reason: None };
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
        PolicyAction::Allow { reason: None }
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

### EventSinkPolicy (generic event producer)

A chainable policy that sends `FsOpContext` to a caller-supplied closure.
The policy does not know what the caller does with the event — it could
log, buffer, or send to a cross-stack correlator channel.

```rust
/// Sends operation context to a caller-supplied sink function.
/// Always returns Allow — observe-only, no enforcement.
pub struct EventSinkPolicy<F> {
    on_event: F,
}

impl<F> EventSinkPolicy<F>
where
    F: Fn(&FsOpContext, Option<&FsOpResult>) + Send + Sync + 'static,
{
    pub fn new(on_event: F) -> Self {
        Self { on_event }
    }
}

impl<F> FsPolicy for EventSinkPolicy<F>
where
    F: Fn(&FsOpContext, Option<&FsOpResult>) + Send + Sync + 'static,
{
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        (self.on_event)(ctx, None);
        PolicyAction::Allow { reason: None }
    }

    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        (self.on_event)(ctx, Some(result));
        PolicyAction::Allow { reason: None }
    }
}
```

Usage at the host runtime level (which depends on both vfs and vnet):

```rust
// Host runtime constructs its own unified event type:
enum RuntimeEvent {
    Fs(motlie_vfs::FsOpContext),
    Net(motlie_vnet::DnsQueryContext),
}

let (tx, rx) = mpsc::channel::<RuntimeEvent>();

// vfs side:
let tx_fs = tx.clone();
let fs_policy = LoggingPolicy
    .chain(CredentialGuard)
    .chain(EventSinkPolicy::new(move |ctx, _| {
        let _ = tx_fs.send(RuntimeEvent::Fs(ctx.clone()));
    }));

// vnet side (analogous EventSinkPolicy in motlie-vnet):
let tx_net = tx.clone();
let net_policy = motlie_vnet::LoggingPolicy
    .chain(motlie_vnet::EventSinkPolicy::new(move |ctx| {
        let _ = tx_net.send(RuntimeEvent::Net(ctx.clone()));
    }));
```

Neither crate depends on the other. The host runtime owns the enum and
the channel. Each crate provides a generic `EventSinkPolicy<F>` that
accepts any `Fn` — it doesn't know about the other crate's types.

## Generic Type Flow

```
FsServer<P: FsPolicy = NoPolicy>
  → dispatch<P>()
    → build FsOpContext (path, tag, overlay status)
    → policy.on_op(&ctx)              // pre-op check
    → if Deny: return errno immediately
    → execute operation
    → build FsOpResult (success, errno, bytes, latency)
    → policy.on_complete(&ctx, &result)  // post-op observe
    → emit_event()                     // enriched FsEvent
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

The `EventSinkPolicy` in both vfs and vnet enables a host-runtime
correlator that sees both filesystem and network activity. This is NOT
a vfs concern — the host runtime owns composition:

- vfs provides: `EventSinkPolicy<F>` + `FsOpContext: Clone + Send`
- vnet provides: `EventSinkPolicy<F>` + context types `Clone + Send`
- Host runtime defines its own `UnifiedEvent` enum wrapping both
- No shared crate needed — each crate is independent

Example correlation signals (host runtime, not vfs):
- "read `/.env` then DNS lookup for `pastebin.com`" → exfiltration
- "write to `/tmp/payload` then connect to unknown IP:4444" → reverse shell

The vfs DESIGN requirement is: **context types must be `Clone + Send`**
so the event sink policy can transport them across thread boundaries.

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
| Chain evaluation (on_op) | Unit: Deny short-circuits, Observe sticky, Allow defers |
| Chain evaluation (on_complete) | Unit: both policies always see completions |
| `LoggingPolicy` | Unit: returns Allow, tracing events match OTel conventions |
| `CredentialGuard` | Unit: denies non-overlay credential paths, allows overlay |
| `EventSinkPolicy` | Unit: closure receives context on both on_op and on_complete |
| `FsOpContext` population | Integration: dispatch populates path, overlay status |
| `FsOpResult` population | Integration: dispatch measures latency, captures bytes |
| `FsEvent` enrichment | Integration: events carry path, bytes, result, latency |
| Zero-cost NoPolicy | Build: binary size delta ≈ 0 vs current code |
| Migration compat | Integration: existing tests pass with NoPolicy default |
