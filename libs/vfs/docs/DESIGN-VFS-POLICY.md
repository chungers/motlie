# motlie-vfs Policy Engine: Enriched Observability and Access Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
    /// Mount-relative path (e.g. "/.ssh/authorized_keys").
    pub path: String,
    /// Whether this path is overlay-managed (injected by the host).
    pub is_overlay: bool,
    /// Timestamp when the operation was received.
    pub timestamp: Instant,
}

/// Result provided to on_complete() after the operation executes.
pub struct FsOpResult {
    /// Whether the operation succeeded.
    pub success: bool,
    /// Errno if the operation failed (0 on success).
    pub errno: i32,
    /// Bytes transferred (for Read/Write operations).
    pub bytes: Option<usize>,
    /// Wall-clock latency of the operation.
    pub latency: Duration,
}
```

`FsOpContext` is `Clone + Send` to support the event sink policy.

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

## Shared Detection Primitives (`motlie-policy`)

Detection utilities (entropy analysis, domain parsing, rate limiters)
live in `libs/policy/` (`motlie-policy` crate) — shared by both
`motlie-vfs` and `motlie-vnet` without either depending on the other.

```toml
# libs/vfs/Cargo.toml
[dependencies]
motlie-policy = { path = "../policy" }
```

For example, a vfs policy that detects high-entropy filenames (potential
encrypted/obfuscated payload staging) can use the same
`motlie_policy::entropy::shannon_entropy()` function used by vnet's
DNS exfiltration detector. The primitives are stack-agnostic.

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
