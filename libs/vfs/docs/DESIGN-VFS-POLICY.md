# motlie-vfs Policy Engine: Enriched Observability and Access Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
/// Callbacks run on the vsock dispatch thread via `&self`. Stateful
/// policies use interior mutability (AtomicU64, RwLock, etc.).
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

### Context Types

```rust
/// Context provided to on_op() before the operation executes.
#[derive(Clone)]
pub struct FsOpContext {
    /// Operation kind.
    pub op: FsOpKind,
    /// Mount tag (e.g. "alice-home").
    pub tag: String,
    /// Inode number of the target file/directory. Enables cross-operation
    /// correlation: a policy can track "read inode X then write inode X"
    /// patterns (in-place encryption detection).
    pub inode: u64,
    /// Mount-relative path (e.g. "/.ssh/authorized_keys").
    pub path: String,
    /// Parent directory path (for Create, Mkdir, Unlink, Rename).
    /// None for operations on existing files (Read, Write, Getattr).
    pub parent_path: Option<String>,
    /// Whether this path is overlay-managed (injected by the host).
    pub is_overlay: bool,
    /// Whether this path matches a predefined sensitive pattern
    /// (.ssh/, .env, credentials, secrets). Pre-classified by the
    /// dispatch layer so policies don't need to recompute it.
    pub is_sensitive: bool,
    /// Timestamp when the operation was received.
    pub timestamp: Instant,
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
    /// None for non-Write operations.
    pub write_entropy: Option<f64>,
    /// Wall-clock latency of the operation.
    pub latency: Duration,
}
```

`FsOpContext` and `FsOpResult` are `Clone + Send` to support the event
sink policy.

**Why `inode` matters:** Filesystem attacks correlate operations on the
same file. A process that reads a file then immediately writes high-entropy
data back to the same inode is likely encrypting it in place (ransomware
pattern). Without inode tracking, the policy can only reason about
individual operations — with it, it can detect sequences.

**Why `write_entropy` is not DPI:** Unlike network payload inspection
(which requires TLS termination), filesystem write data passes through
the vfs server in plaintext — the server already copies it from the FUSE
buffer to disk. Computing entropy of the buffer before writing is
essentially free (~50ns) and provides a strong signal for detecting
encrypted/compressed staging without inspecting the actual content
semantics.

**Why `is_sensitive` is pre-classified:** Every policy in the chain
would otherwise need to implement the same path-matching logic for
credential paths. Pre-classifying in the dispatch layer avoids redundant
work and ensures consistent classification across policies.

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

**Example: ScanDetector**

```rust
struct ScanDetector {
    // tag → (op_count, unique_paths, window_start)
    windows: RwLock<HashMap<String, (u64, HashSet<String>, Instant)>>,
}

impl FsPolicy for ScanDetector {
    fn on_op(&self, ctx: &FsOpContext) -> PolicyAction {
        if !matches!(ctx.op, FsOpKind::Lookup | FsOpKind::Getattr | FsOpKind::Readdir) {
            return PolicyAction::Allow { reason: None };
        }
        let mut windows = self.windows.write().unwrap();
        let entry = windows.entry(ctx.tag.clone())
            .or_insert((0, HashSet::new(), Instant::now()));
        if entry.2.elapsed() > Duration::from_secs(10) {
            *entry = (0, HashSet::new(), Instant::now());
        }
        entry.0 += 1;
        entry.1.insert(ctx.path.clone());

        // >500 metadata ops or >200 unique paths in 10s = scan
        if entry.0 > 500 || entry.1.len() > 200 {
            PolicyAction::Observe {
                reason: PolicyReason::Custom(
                    format!("scan pattern: {} ops, {} paths in 10s",
                            entry.0, entry.1.len()).into()
                ),
            }
        } else {
            PolicyAction::Allow { reason: None }
        }
    }
}
```

**API coverage:** Fully supported. `FsOpContext` provides `op`, `path`,
`timestamp`, `is_sensitive`, `is_overlay`. All discovery patterns are
detectable with stateful `on_op()` policies.

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

**Example: RansomwareDetector**

```rust
struct RansomwareDetector {
    // inode → (last_read_bytes, last_read_time)
    recent_reads: RwLock<HashMap<u64, (usize, Instant)>>,
}

impl FsPolicy for RansomwareDetector {
    fn on_complete(&self, ctx: &FsOpContext, result: &FsOpResult) -> PolicyAction {
        match ctx.op {
            FsOpKind::Read => {
                if let Some(bytes) = result.bytes {
                    self.recent_reads.write().unwrap()
                        .insert(ctx.inode, (bytes, Instant::now()));
                }
            }
            FsOpKind::Write => {
                if let (Some(write_bytes), Some(entropy)) = (result.bytes, result.write_entropy) {
                    let reads = self.recent_reads.read().unwrap();
                    if let Some((read_bytes, read_time)) = reads.get(&ctx.inode) {
                        // Read N bytes, then write ~N high-entropy bytes
                        // to the same inode within 1 second = encryption
                        let similar_size = (*read_bytes as f64 - write_bytes as f64).abs()
                            / (*read_bytes as f64).max(1.0) < 0.2;
                        let recent = read_time.elapsed() < Duration::from_secs(1);
                        if similar_size && recent && entropy > 4.0 {
                            return PolicyAction::Deny {
                                errno: libc::EACCES,
                                reason: PolicyReason::Custom(
                                    format!("ransomware pattern: read {}B then write {}B \
                                             (entropy {:.1}) to inode {}",
                                            read_bytes, write_bytes, entropy, ctx.inode).into()
                                ),
                            };
                        }
                    }
                }
            }
            _ => {}
        }
        PolicyAction::Allow { reason: None }
    }
}
```

**API coverage:** Fully supported. `ctx.inode` enables cross-operation
correlation. `result.write_entropy` detects encryption. Stateful policy
tracks read→write sequences per inode. No additional callbacks needed —
`on_complete()` with the enriched context is sufficient.

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
