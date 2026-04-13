# motlie-vnet Policy Engine: Observability and Egress Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-05 | @claude-tl | Full construction code, correlator event handling with confidence values, example event stream, guest quarantine (future TODO) |
| 2026-04-05 | @claude-tl | Cross-stack correlator design: unified event pipeline, credential→exfil walkthrough, staging→execution walkthrough, ML/LLM as correlator, architecture diagram, PID preconditions |
| 2026-04-05 | @claude-tl | PolicyEvaluator plumbing: builder→config→slirp thread, chaining examples with walkthroughs (known-bad, unknown, known-good) |
| 2026-04-05 | @claude-tl | Drop Observe, two-action PolicyAction (Allow/Deny + confidence), deny_threshold, emission truth table |
| 2026-04-05 | @claude-tl | Network metadata gaps evaluated: keep DNS counts + guest PID, reject TTL/TCP options/MAC/libslirp state |
| 2026-04-05 | @claude-tl | Confidence on PolicyAction, InterceptPolicy with ArcSwap (latency analysis), PolicyEvent enum at call site |
| 2026-04-05 | @claude-tl | Connection timeline: 3-phase detection (DNS intent → handshake → flow), technique tables, JA3/beacon gaps |
| 2026-04-05 | @claude-tl | DNS exfiltration detection, entropy analysis, label_lengths/name_wire_len in DnsQueryContext |
| 2026-04-05 | @claude-tl | errno on PolicyAction, predefined errno pairs, OTel logging, cross-stack alignment with vfs |
| 2026-04-05 | @claude-tl | PolicyReason (Cow-backed enum), on_tcp_flow with flow metadata + SNI, DPI limitations |
| 2026-04-05 | @claude-tl | Redesign: generic policy (no Box/dyn), chainable via trait, NoPolicy default, intent derivation layering |
| 2026-04-05 | @claude-tl | Initial DESIGN: DNS interception, TCP connection control, intent-based policy, callback API |

## Problem Statement

Guest VMs with outbound internet access (via `motlie-vnet`) can reach any
host on the internet. For a development-oriented platform like motlie, this
is both a feature (install packages, clone repos, call APIs) and a risk
(data exfiltration, supply chain attacks, unexpected dependencies on
external services).

The host runtime needs visibility into what the guest is doing on the
network and the ability to enforce policy — but blanket allow/deny lists
are too coarse. A developer's `apt update` and `git clone` should work
seamlessly, while connections to ad networks, analytics beacons, or
suspicious domains should be caught and optionally blocked.

### Goals

1. **Observability**: log every DNS lookup and TCP connection attempt with
   enough context (domain name, resolved IP, source port, timestamp) for
   the host runtime to reason about guest network behavior.

2. **Policy control**: allow the host runtime to evaluate each DNS query
   and TCP connection against a policy and decide: allow, deny (with
   logged reason). Confidence (0.0–1.0) on every decision enables a
   configurable `deny_threshold` that auto-converts uncertain Allows
   to Denys.

3. **Intent-based policy**: policy decisions should consider the *intent*
   behind an action, not just the domain name. The engine provides
   context; the policy callback decides. See "Intent Derivation" for
   what context is available at each stage.

4. **Extensibility**: the policy engine is a Rust trait with a `chain()`
   method for composition. Policies are generic type parameters —
   monomorphized at compile time with zero dynamic dispatch.

5. **Zero-cost when unused**: when no policy is configured, the interceptor
   is compiled out entirely. The hot path is identical to no-policy code.

### Non-Goals

- Deep packet inspection (DPI) — we inspect headers, not payload content
- TLS/HTTPS termination or MITM — out of scope
- Bandwidth throttling or traffic shaping
- Stateful application-layer protocol parsing (e.g. HTTP method inspection)
- Policy persistence or management UI (the host runtime owns this)

## Architecture

### Interception Points

libslirp is a C library with no built-in policy hooks. All interception
happens at the Rust frame level — before frames enter libslirp (TX path)
or after libslirp produces response frames (RX path).

```
Guest VM
  │
  ▼
┌─────────────────────────────────────────────────┐
│ TX path (guest → internet)                      │
│                                                   │
│ process_tx() extracts Ethernet frame from vring │
│        │                                         │
│        ▼                                         │
│  ┌─────────────────────────────────────┐        │
│  │ TX Interceptor<P: EgressPolicy>     │        │
│  │                                     │        │
│  │ Parse: Eth → IP → UDP/TCP headers  │        │
│  │                                     │        │
│  │ DNS query (UDP port 53)?           │        │
│  │   → extract queried domain name    │        │
│  │   → call policy.on_dns_query()     │        │
│  │   → if Deny: forge NXDOMAIN, skip │        │
│  │     slirp.input(), inject response │        │
│  │                                     │        │
│  │ TCP SYN?                           │        │
│  │   → extract dst IP:port            │        │
│  │   → call policy.on_tcp_connect()   │        │
│  │   → if Deny: forge RST, skip      │        │
│  │     slirp.input(), inject RST      │        │
│  │                                     │        │
│  │ Otherwise: pass through            │        │
│  └─────────────────────────────────────┘        │
│        │                                         │
│        ▼                                         │
│  slirp.input(frame) → libslirp → host sockets  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ RX path (internet → guest)                      │
│                                                   │
│ libslirp → send_packet() → rx_queue            │
│        │                                         │
│        ▼                                         │
│  ┌─────────────────────────────────────┐        │
│  │ RX Interceptor<P: EgressPolicy>     │        │
│  │                                     │        │
│  │ DNS response (UDP port 53)?        │        │
│  │   → parse response, extract domain │        │
│  │     + resolved IPs                  │        │
│  │   → call policy.on_dns_response()  │        │
│  │   → update domain→IP reverse map  │        │
│  │                                     │        │
│  │ Otherwise: pass through            │        │
│  └─────────────────────────────────────┘        │
│        │                                         │
│        ▼                                         │
│  process_rx() → inject into guest rx vring      │
└─────────────────────────────────────────────────┘
```

### Why TX-Side DNS Interception

Denying DNS on the TX side (before `slirp.input()`) is preferred because:
- The query never reaches the host network — no DNS leakage
- We can forge a response (NXDOMAIN) and inject it directly into the
  guest's rx path without libslirp involvement
- The guest sees a normal DNS failure, not a timeout

RX-side DNS interception is used for **observation** (logging resolved IPs,
building the domain→IP reverse map) and as a secondary enforcement point.

### Why TX-Side TCP Interception

Denying TCP on the TX side (before `slirp.input()`) prevents libslirp
from creating a host-side socket. Benefits:
- No host-side connect() syscall — zero network footprint for denied connections
- We can forge a TCP RST and inject it so the guest sees immediate rejection
- The domain→IP reverse map (populated from DNS responses) enriches TCP
  decisions with domain context even though TCP only has IP addresses

## Policy API

### PolicyReason

Policy reasons are zero-allocation for common cases (category labels)
and flexible for custom policies:

```rust
/// Reason attached to a policy decision. Zero-alloc for categories,
/// Cow-backed for custom strings.
pub enum PolicyReason {
    /// Domain category (zero-alloc — the category is a static label).
    Category(DomainCategory),
    /// Custom reason string. Static for compile-time constants,
    /// owned for runtime-generated reasons.
    Custom(Cow<'static, str>),
}

// Convenience: any &'static str or String converts to PolicyReason
impl From<&'static str> for PolicyReason { ... }
impl From<String> for PolicyReason { ... }
impl From<DomainCategory> for PolicyReason { ... }
```

### Core Trait

```rust
/// Policy decision. Binary: Allow or Deny. Confidence carries the
/// uncertainty signal — no separate "Observe" variant needed.
/// Shared shape with motlie-vfs's FsPolicy.
pub enum PolicyAction {
    /// Allow the request to proceed.
    Allow {
        /// Why this was allowed. None = no notable reason (confident allow).
        /// The chain keeps the reason from the lowest-confidence policy
        /// (the weakest link explains why it's weak).
        reason: Option<PolicyReason>,
        /// Confidence in this decision (0.0–1.0).
        /// 1.0 = certain (known-good domain).
        /// 0.5 = uncertain (unknown domain).
        /// 0.1 = barely allowed (high entropy, almost blocked).
        /// Below the deny_threshold → converted to Deny at the call site.
        confidence: f64,
    },
    /// Deny the request. The interceptor forges a failure response
    /// (NXDOMAIN, RST) based on errno.
    Deny {
        errno: i32,
        reason: PolicyReason,
        /// Confidence in the denial.
        /// 1.0 = definitive (known-bad, blocklist match).
        /// Lower values = policy-decided deny with some uncertainty.
        confidence: f64,
    },
}
```

**Chain evaluation:**
- First `Deny` wins — its reason and confidence are the composite.
- All `Allow` → composite is `Allow` with the **minimum confidence**
  across the chain. The **reason comes from the policy that had the
  lowest confidence** (the weakest link explains *why* it's weak).

**Example chain evaluation:**

```
Chain: LoggingPolicy → CategoryPolicy → ExfilDetector

LoggingPolicy.on_dns_query(ctx)  → Allow { reason: None, confidence: 1.0 }
CategoryPolicy.on_dns_query(ctx) → Allow { reason: Category(Unknown), confidence: 0.5 }
ExfilDetector.on_dns_query(ctx)  → Allow { reason: Custom("moderate entropy"), confidence: 0.3 }

Composite: Allow {
    reason: Some(Custom("moderate entropy")),  // from ExfilDetector (0.3 = lowest)
    confidence: 0.3,
}
```

**deny_threshold:** The call site converts low-confidence Allows to Denys:

```rust
// In VnetConfig:
let config = VnetConfig::builder()
    .socket_path("/tmp/vnet.sock")
    .egress_policy(my_chain)
    .deny_threshold(0.5)  // Allow with confidence < 0.5 → Deny
    .build()?;
```

Default threshold: `0.0` (only explicit Denys block). Strict mode: `0.8`.

**Hot-path before and after:**

Before (current `slirp_thread_main`, no policy):
```rust
while let Ok(frame) = tx_receiver.try_recv() {
    slirp.input(&frame);   // every frame passes through, no inspection
}
```

After (with policy + threshold + emission):
```rust
while let Ok(frame) = tx_receiver.try_recv() {
    match parse_frame(&frame) {
        ParsedPacket::DnsQuery(ctx) => {
            let action = policy.on_dns_query(&ctx);          // ~50ns (chain)
            let action = apply_threshold(action, threshold);  // ~5ns (comparison)
            emit_if_interesting(&event_tx, &ctx, &action);   // ~20ns (try_send or skip)
            match action {
                PolicyAction::Deny { errno, .. } => {
                    let resp = forge_nxdomain(&frame, errno);
                    rx_sender.send(resp);                     // forged response to guest
                }
                PolicyAction::Allow { .. } => slirp.input(&frame),
            }
        }
        ParsedPacket::TcpSyn(ctx) => { /* same pattern with on_tcp_connect */ }
        ParsedPacket::Other => slirp.input(&frame),  // non-DNS/TCP: pass through
    }
}
```

When `P = NoPolicy` and `threshold = 0.0`, the compiler eliminates
`parse_frame`, `apply_threshold`, `emit_if_interesting`, and the match
arms — the loop compiles to the same code as "before."

**Call site logic (detailed):**

```rust
// Step 1: call the policy chain
let action = policy.on_dns_query(&ctx);

// Step 2: apply threshold
let action = match action {
    PolicyAction::Allow { confidence, reason } if confidence < deny_threshold => {
        PolicyAction::Deny {
            errno: libc::EHOSTUNREACH,
            reason: reason.unwrap_or(PolicyReason::Custom("below threshold".into())),
            confidence,
        }
    }
    other => other,
};

// Step 3: emit (always Deny, uncertain Allow only)
match &action {
    PolicyAction::Deny { .. } => {
        event_tx.try_send(PolicyEvent::DnsQuery { ctx: ctx.clone(), action: action.clone() });
    }
    PolicyAction::Allow { confidence, .. } if *confidence < 1.0 => {
        event_tx.try_send(PolicyEvent::DnsQuery { ctx: ctx.clone(), action: action.clone() });
    }
    _ => {}  // confident Allow — skip (not interesting)
};

// Step 4: enforce
match &action {
    PolicyAction::Deny { errno, .. } => inject_nxdomain(&frame, *errno),
    PolicyAction::Allow { .. } => slirp.input(&frame),
}
```

**Emission truth table:**

| Chain result | Threshold | After threshold | Emitted? | Correlator sees |
|---|---|---|---|---|
| Allow { 1.0 } | 0.5 | Allow { 1.0 } | No | Nothing (confident, not interesting) |
| Allow { 0.9 } | 0.5 | Allow { 0.9 } | Yes | Uncertain allow — worth watching |
| Allow { 0.3 } | 0.5 | **Deny { 0.3 }** | Yes | Threshold-converted deny |
| Deny { 0.95 } | 0.5 | Deny { 0.95 } | Yes | Policy-decided deny |

**Confidence use cases:**

| Policy type | Confidence source | Example |
|------------|------------------|---------|
| Rule-based (`CategoryPolicy`) | 1.0 (certain match) | `Allow { Category(PackageManager), 1.0 }` |
| Heuristic (`ExfilDetector`) | 0.3–0.8 (signal strength) | `Allow { Custom("moderate entropy"), 0.3 }` |
| Blocklist (`InterceptPolicy`) | 1.0 (correlator decided) | `Deny { Custom("correlator: blocked"), 1.0 }` |
| ML model (future) | Model probability | `Allow { Custom("model: benign"), 0.87 }` |

### Predefined (errno, reason) Pairs for Network Denials

| errno | DomainCategory | Forged response | Use case |
|-------|---------------|-----------------|----------|
| `EHOSTUNREACH` | `AdNetwork` | DNS NXDOMAIN | Blocked domain lookup |
| `ENETUNREACH` | `Analytics` | DNS NXDOMAIN | Blocked analytics domain |
| `ECONNREFUSED` | `Unknown` | TCP RST | Denied TCP connection |
| `EACCES` | (any) | TCP RST | Generic policy denial |

The interceptor maps errno to the forged response:
- `EHOSTUNREACH`, `ENETUNREACH` → forge DNS NXDOMAIN response
- `ECONNREFUSED`, `EACCES` → forge TCP RST

```

/// Egress policy callbacks. All methods have default allow-all impls.
///
/// Callbacks run on the slirp thread via `&self`. They must not block
/// or perform I/O — compute a decision and return immediately.
///
/// **Statefulness**: the trait uses `&self` to keep chaining simple.
/// Policies that need mutable state (counters, dynamic allowlists)
/// should use interior mutability (`AtomicU64`, `RwLock`, etc.). The
/// policy implementer manages their own synchronization — the trait
/// does not impose a synchronization strategy.
///
/// For async policy evaluation, return `Allow` with low confidence and
/// asynchronously with a later connection teardown if needed.
pub trait EgressPolicy: Send + 'static {
    /// Called when the guest issues a DNS query. Return Deny to prevent
    /// the query from reaching the host resolver (forged NXDOMAIN).
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    /// Called when a DNS response arrives from the host resolver,
    /// before it is forwarded to the guest.
    fn on_dns_response(&self, ctx: &DnsResponseContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    /// Called when the guest initiates a TCP connection (SYN detected).
    /// Return Deny to forge a RST — no host-side socket is created.
    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    /// Called periodically for active TCP flows and once on flow close.
    /// Provides flow-level metadata: byte counts, timing, SNI.
    ///
    /// **Cannot see encrypted payload.** For HTTPS connections (the
    /// majority of traffic), only metadata is visible — see
    /// "Flow-Level Observability" for what is and is not available.
    ///
    /// Return Deny to tear down the flow (forge RST in both directions).
    fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    /// Chain another policy after this one.
    ///
    /// Evaluation order: `self` first, then `next`.
    /// - First `Deny` wins (short-circuit).
    /// - Minimum confidence propagated across the chain.
    /// - `Allow` defers to `next`.
    fn chain<N: EgressPolicy>(self, next: N) -> impl EgressPolicy
    where
        Self: Sized;
}
```

### Chaining

Policies compose via `.chain()`. The developer never names or sees the
intermediate types — the compiler resolves them.

```rust
let policy = LoggingPolicy
    .chain(InterceptPolicy::new(blocked.clone()))
    .chain(CategoryPolicy::default())
    .chain(DnsExfilDetector::new(config));

// `policy` implements EgressPolicy. No heap allocation, no vtable dispatch.
```

Chain evaluation rules:
- First `Deny` wins (short-circuit)
- Minimum confidence propagated across the chain (weakest link)
- `Allow` defers to next policy
- Minimum confidence propagated across the chain

### Zero-Cost NoPolicy Default

```rust
/// Zero-cost policy that allows everything. When used as the policy type,
/// the compiler optimizes the entire interceptor away — the hot path
/// compiles to the same code as if policy support did not exist.
pub struct NoPolicy;

impl EgressPolicy for NoPolicy {
    fn on_dns_query(&self, _: &DnsQueryContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
    fn on_dns_response(&self, _: &DnsResponseContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
    fn on_tcp_connect(&self, _: &TcpConnectContext) -> PolicyAction {
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

### InterceptPolicy (externally-fed blocklist)

A policy whose deny set is managed by an external correlator thread.
The policy itself is stateless in its decision logic — it reads from a
concurrent snapshot, never writes. The correlator writes.

```rust
use arc_swap::ArcSwap;

pub struct InterceptPolicy {
    /// Blocklist snapshot. Updated atomically by the correlator thread.
    /// Policy reads via load() — ~5ns, zero contention.
    blocked: Arc<ArcSwap<HashSet<(Ipv4Addr, u16)>>>,
}

impl EgressPolicy for InterceptPolicy {
    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        let snapshot = self.blocked.load();
        if snapshot.contains(&(ctx.dst_ip, ctx.dst_port)) {
            PolicyAction::Deny {
                errno: libc::ECONNREFUSED,
                reason: PolicyReason::Custom("correlator: blocked".into()),
                confidence: 1.0,  // correlator already decided
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
current.insert((target_ip, target_port));
blocklist.store(Arc::new(current));
```

**Latency analysis — why ArcSwap:**

| Approach | Hot-path read | Correlator write | Contention |
|----------|-------------|-----------------|-----------|
| `Arc<RwLock<HashSet>>` | ~50ns (read lock + lookup + release) | ~100ns (write lock) | Possible: writer blocks readers |
| `try_recv` + local map | ~50ns (try_recv miss + local lookup) | ~20ns (channel send) | None, but 1-call delay |
| **`ArcSwap<HashSet>`** | **~25ns** (atomic load + lookup) | ~50ns + O(N) clone per update | **Zero: readers never block** |

ArcSwap wins: 2x faster reads than RwLock, zero reader-writer contention.
The O(N) clone happens on the correlator thread (off hot path) and only
when the blocklist changes (rare — seconds between updates, not per-packet).
For a 1000-entry blocklist, clone is ~50μs — negligible for a thread that
runs every few seconds.

**Feedback loop:**

```
Slirp thread (hot path)         Correlator thread (slow path)
    │                                    │
    ├─ policy.on_dns_query(&ctx)        │
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

### Logging as a Policy (OTel-compatible)

Logging is not special — it's a regular `EgressPolicy` implementation
that always returns `Allow` but emits OTel-compatible tracing events.
If the host runtime wires `tracing-opentelemetry`, events automatically
flow to Jaeger/OTLP without policy changes.

```rust
pub struct LoggingPolicy;

impl EgressPolicy for LoggingPolicy {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "CLIENT",
            otel.name = "vnet.dns.query",
            net.host.name = %ctx.domain,
            net.protocol.name = "dns",
            net.source.port = ctx.source_port,
            "net.dns.start"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "CLIENT",
            otel.name = "vnet.tcp.connect",
            net.peer.ip = %ctx.dst_ip,
            net.peer.port = ctx.dst_port,
            net.host.name = ?ctx.domain,
            net.source.port = ctx.source_port,
            "net.tcp.start"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "CLIENT",
            otel.name = "vnet.tcp.flow",
            net.peer.ip = %ctx.dst_ip,
            net.peer.port = ctx.dst_port,
            net.host.name = ?ctx.domain,
            net.tls.sni = ?ctx.sni,
            net.bytes.tx = ctx.bytes_tx,
            net.bytes.rx = ctx.bytes_rx,
            net.duration_ms = ctx.duration.as_millis() as u64,
            net.flow.closed = ctx.is_close,
            "net.tcp.flow"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

### Policy Context Types

```rust
pub struct DnsQueryContext {
    /// The queried domain name (e.g. "archive.ubuntu.com").
    pub domain: String,
    /// DNS record type (A, AAAA, CNAME, etc.).
    pub query_type: DnsQueryType,
    /// Guest-side source port.
    pub source_port: u16,
    /// Timestamp.
    pub timestamp: Instant,
    /// Individual label lengths in the queried name.
    /// e.g. "aGVsbG8.exfil.com" → [7, 5, 3].
    /// Useful for detecting DNS exfiltration (encoded payloads in
    /// long subdomain labels). See "DNS Exfiltration Detection".
    pub label_lengths: Vec<usize>,
    /// Total wire-format length of the DNS name (bytes, including
    /// length prefixes and root label). Exceeding ~200 bytes is
    /// a strong exfiltration signal.
    pub name_wire_len: usize,
}

pub struct DnsResponseContext {
    /// The queried domain name.
    pub domain: String,
    /// Resolved IPv4 addresses.
    pub resolved_ips: Vec<Ipv4Addr>,
    /// TTL from the DNS response.
    pub ttl: u32,
    /// Timestamp.
    pub timestamp: Instant,
}

pub struct TcpConnectContext {
    /// Destination IP address.
    pub dst_ip: Ipv4Addr,
    /// Destination port.
    pub dst_port: u16,
    /// Domain name that resolved to this IP, if known from a prior DNS
    /// lookup. None if the guest connected directly by IP.
    pub domain: Option<String>,
    /// Guest-side source port.
    pub source_port: u16,
    /// Timestamp.
    pub timestamp: Instant,
    /// Guest process name that initiated this connection, if reported
    /// by the guest agent via vsock. None until russhd integration (v2).
    pub guest_process: Option<String>,
}

pub struct TcpFlowContext {
    /// Destination IP address.
    pub dst_ip: Ipv4Addr,
    /// Destination port.
    pub dst_port: u16,
    /// Domain name from DNS reverse map (if resolved via DNS).
    pub domain: Option<String>,
    /// TLS Server Name Indication extracted from the ClientHello.
    /// Available on the first TX packet of a TLS connection (plaintext
    /// before encryption starts). Provides the domain even if the guest
    /// bypassed DNS and connected by IP. None for non-TLS connections
    /// or if the ClientHello did not contain SNI.
    pub sni: Option<String>,
    /// Guest-side source port.
    pub source_port: u16,
    /// Cumulative bytes sent guest→host.
    pub bytes_tx: u64,
    /// Cumulative bytes sent host→guest.
    pub bytes_rx: u64,
    /// Time since the TCP SYN.
    pub duration: Duration,
    /// Whether this is the final report (flow closed).
    pub is_close: bool,
    /// Timestamp.
    pub timestamp: Instant,
    /// Guest process name (v2, requires russhd).
    pub guest_process: Option<String>,
}
```

## Intent Derivation

Intent-based policy requires context about *why* a connection is
happening, not just *where* it goes. The available context depends on
the maturity of the guest agent integration:

### v1: Domain + Port Heuristics (available now)

The policy engine can classify connections based on:
- **Domain patterns**: suffix matching (`*.ubuntu.com` → PackageManager)
- **Well-known ports**: 443 (HTTPS), 22 (SSH), 80 (HTTP)
- **Destination port + domain combination**: port 443 + `crates.io` = cargo fetch

This is the CategoryPolicy's approach. It's a heuristic — better than
blanket IP rules, but cannot distinguish `curl evil.com` from `apt update`
when both use HTTPS to the same CDN.

### v2: Guest Process Identity (requires russhd, Phase 3.6)

When the host-side russhd + vsock agent is running, the guest agent can
report which PID opened which socket. This flows into `TcpConnectContext`
and `TcpFlowContext` via the `guest_process: Option<String>` field
(already present in the context types — `None` until the agent is live).

Enables true intent-based decisions:
- `apt` connecting to `ubuntu.com` → Allow (PackageManager + known process)
- `unknown-binary` connecting to `ubuntu.com` → Allow { confidence: 0.3 } (unexpected process)

### v3: Project Manifest Correlation (future)

The host runtime could parse `Cargo.toml`, `package.json`, `requirements.txt`
to derive an expected dependency set, then auto-generate an allowlist of
domains those dependencies need (crates.io, npmjs.com, pypi.org, etc.).
Connections outside the manifest are flagged.

This is a host-runtime concern, not a vnet concern — it would be
implemented as a custom `EgressPolicy` that reads the manifest.

## Flow-Level Observability

The `on_tcp_flow` callback provides metadata about active and closed TCP
connections. This is distinct from `on_tcp_connect` (which fires once at
SYN time) — `on_tcp_flow` is called periodically during the connection
lifetime and once on close.

### What Is Visible

| Signal | Source | When |
|--------|--------|------|
| Flow identity (src/dst IP:port) | IP/TCP headers | Every report |
| Domain name | DNS reverse map | If resolved via DNS |
| TLS SNI (Server Name Indication) | ClientHello first byte (plaintext) | First TX packet only |
| Byte counts (tx/rx) | Cumulative frame sizes | Every report |
| Duration since SYN | Timer | Every report |
| Flow close event | TCP FIN/RST detection | Once |
| Guest process name | vsock agent (v2) | When available |

### What Is NOT Visible

| Signal | Why |
|--------|-----|
| HTTP method, path, headers | Encrypted after TLS handshake |
| Request/response body | Encrypted |
| Application-layer protocol | Cannot infer from encrypted bytes |
| Certificate details | Would require TLS parsing beyond SNI |
| Individual request boundaries | Encrypted; HTTP/2 multiplexes |

## Anomaly Detection by Connection Phase

A network connection is a timeline. Detection moves from identifying
the **intent** (DNS), to the **handshake** (pre-connection/initial), and
finally to the **behavior** (flow/established). The vnet policy API
provides callbacks at each phase — the right detector runs at the right
stage.

### Phase 1: DNS — The "Intent" Phase

The connection hasn't started. The guest is asking "where is this?"
This is the best place to catch stealthy smuggling before a single byte
of real data is exchanged.

**Policy callback:** `on_dns_query(&self, ctx: &DnsQueryContext)`

| Technique | Method | What it Detects | API fields used |
|-----------|--------|----------------|-----------------|
| **Entropy Analysis** | Calculate Shannon entropy of subdomain labels | High-entropy strings encoding data (`a1z9b2k8...`) | `ctx.domain`, `ctx.label_lengths` → `policy::entropy::shannon_entropy()` |
| **Request Frequency** | Monitor DNS query volume per base domain over time | High-volume "chunking" — data split across thousands of queries | `ctx.domain`, `ctx.timestamp` → stateful rate counter (`RwLock<HashMap>`) |
| **Domain Aging** | Check registration date of destination domain | "Burner" domains registered recently for the attack | `ctx.domain` → external WHOIS/threat-intel lookup (async, return Allow with low confidence) |
| **Record Type Profiling** | Monitor for rare record types (TXT, NULL, CNAME chains) | Non-standard DNS records carrying larger data payloads | `ctx.query_type` → flag non-A/AAAA lookups to unknown domains |

**API coverage:** Entropy and frequency are fully supported with existing
context fields. Domain aging requires external lookup (policy returns
Allow with low confidence, resolves asynchronously). Record type profiling uses
`ctx.query_type` which is already in `DnsQueryContext`.

### Phase 2: Pre-Connection / Initial Handshake

The TCP 3-way handshake and TLS negotiation. Looking at the identity
and parameters of the connection before data flows.

**Policy callbacks:** `on_tcp_connect(&self, ctx: &TcpConnectContext)`,
`on_tcp_flow(&self, ctx: &TcpFlowContext)` (first report with SNI)

| Technique | Method | What it Detects | API fields used |
|-----------|--------|----------------|-----------------|
| **JA3/JA3S Fingerprinting** | Analyze TLS ClientHello cipher suites and extensions | Known malicious tools (Python/Rust exfiltrators with non-browser signatures) | `TcpFlowContext.sni` + future: raw ClientHello bytes for JA3 hash |
| **Protocol Validation** | Verify traffic on a port matches expected protocol | Protocol tunneling — SSH or raw data over port 443 | First TX bytes in `on_tcp_flow` — check for TLS record header vs raw data |
| **Geo-Fencing / Rare IP** | Compare destination IP against historical baselines | Connections to rare ASNs or countries where the guest never talks | `ctx.dst_ip` → external GeoIP lookup, or stateful IP baseline |
| **SNI / DNS Mismatch** | Compare TLS SNI with DNS-resolved domain | Domain fronting — hiding the real destination behind a CDN | `TcpFlowContext.sni` vs `TcpFlowContext.domain` |

**API coverage:** SNI extraction and DNS/SNI mismatch are fully supported.
JA3 fingerprinting would require exposing raw ClientHello bytes in
`TcpFlowContext` (not currently available — future enhancement). Protocol
validation can check first-byte patterns in `on_tcp_flow`. GeoIP requires
external lookup.

**API gap for JA3:** The current SNI extraction parses only the server
name extension. Full JA3 fingerprinting needs the cipher suite list,
extension list, and elliptic curve parameters from the ClientHello. This
could be added to `TcpFlowContext` as an optional `ja3_hash: Option<String>`
field, computed during SNI extraction at near-zero additional cost (the
ClientHello is already being parsed).

### Phase 3: Flow — The "Established" Session

The connection is live and data is moving. Analyze the shape and rhythm
of the traffic.

**Policy callback:** `on_tcp_flow(&self, ctx: &TcpFlowContext)`

| Technique | Method | What it Detects | API fields used |
|-----------|--------|----------------|-----------------|
| **Directional Asymmetry** | Measure bytes_tx / bytes_rx ratio | Large-scale uploads from a machine that is normally a consumer | `ctx.bytes_tx`, `ctx.bytes_rx` → `policy::ratio` |
| **Duration Analysis** | Identify long-lived sessions with low throughput | Persistent tunnels trickling data out slowly over hours | `ctx.duration`, `ctx.bytes_tx` → bytes/sec rate |
| **Beaconing Detection** | Find fixed timing intervals between reports | Automated heartbeats — malware checking in at exact intervals (e.g. every 30s) | `ctx.timestamp` → stateful inter-report timing, coefficient of variation |
| **Payload Entropy** | Measure randomness of the data stream | Distinguishing natural encrypted traffic from high-entropy data dumps | Not directly available — would require sampling payload bytes (DPI concern) |

**API coverage:** Directional asymmetry and duration analysis are fully
supported. Beaconing detection is partially supported (requires stateful
inter-report timing; report interval is not policy-configurable — see API
gap below). Payload entropy conflicts with the No-DPI design constraint
and is explicitly out of scope.

**API gap for beaconing:** The `on_tcp_flow` report interval is
interceptor-controlled, not policy-controlled. If reports are every 10s
but the beacon interval is 30s, the policy sees 3 reports per beacon
cycle — sufficient for detection. But if reports are every 60s and the
beacon is every 30s, periodicity is masked. Future enhancement: make
the report interval configurable per-flow or per-policy.

### Phase Summary

```
Guest DNS query        TCP SYN            TLS ClientHello        Data flow
     │                    │                     │                    │
     ▼                    ▼                     ▼                    ▼
┌──────────┐      ┌──────────────┐      ┌──────────────┐    ┌──────────────┐
│ Phase 1  │      │   Phase 2    │      │   Phase 2    │    │   Phase 3    │
│ DNS      │      │ TCP Connect  │      │ TLS Hello    │    │ Flow         │
│          │      │              │      │              │    │              │
│ Entropy  │      │ Geo-fence    │      │ JA3 (future) │    │ Byte ratio   │
│ Rate     │      │ Rare IP      │      │ SNI verify   │    │ Duration     │
│ Aging    │      │ Protocol val │      │ Protocol val │    │ Beaconing    │
│ Type     │      │              │      │              │    │              │
│          │      │              │      │              │    │              │
│ callback:│      │ callback:    │      │ callback:    │    │ callback:    │
│ on_dns_  │      │ on_tcp_      │      │ on_tcp_      │    │ on_tcp_      │
│ query()  │      │ connect()    │      │ flow() #1    │    │ flow() #N    │
└──────────┘      └──────────────┘      └──────────────┘    └──────────────┘
```

### Use Cases for Flow Metadata

**Exfiltration detection**: a flow to an Unknown domain that transfers
>10MB in the TX direction is suspicious. The policy can flag or tear it
down:

```rust
fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
    if ctx.bytes_tx > 10_000_000 {
        if let Some(domain) = &ctx.domain {
            if self.categorize(domain) == DomainCategory::Unknown {
                return PolicyAction::Deny {
                    reason: PolicyReason::Custom(
                        format!("large upload to unknown domain: {} ({} bytes)",
                                domain, ctx.bytes_tx).into()
                    ),
                };
            }
        }
    }
    PolicyAction::Allow { reason: None, confidence: 1.0 }
}
```

**C2 beacon detection**: a flow with periodic small tx/rx bursts to an
Unknown domain over a long duration (>30min) matches a command-and-control
pattern. The policy returns `Allow { confidence: 0.2 }` — the
deny_threshold or correlator decides whether to block.

**Bandwidth accounting**: track bytes per domain/category for reporting.
`LoggingPolicy` can aggregate flow stats and emit periodic summaries.

**SNI verification**: if the DNS reverse map says domain X but the TLS
SNI says domain Y, the connection may be using domain fronting. The
policy can flag the mismatch:

```rust
fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
    if let (Some(dns_domain), Some(sni)) = (&ctx.domain, &ctx.sni) {
        if dns_domain != sni {
            return PolicyAction::Allow {
                reason: Some(PolicyReason::Custom(
                    format!("SNI mismatch: DNS={}, SNI={}", dns_domain, sni).into()
                )),
                confidence: 0.2,  // strong suspicion but allow (threshold decides)
            };
        }
    }
    PolicyAction::Allow { reason: None, confidence: 1.0 }
}
```

### DNS Exfiltration Detection

DNS exfiltration encodes stolen data in subdomain labels of DNS queries:

```
aGVsbG8gd29ybGQ.chunk2.attacker.com    ← base64 payload in labels
78e731027d8fd50ed642340b7c9a63b3.x.evil.io  ← hex-encoded data
```

The attacker controls the authoritative nameserver for `attacker.com` and
receives the encoded data as query labels — bypassing all TCP-level
egress controls because DNS queries appear to go to the configured
resolver (`10.0.2.3`), not to the attacker directly.

**This is a DNS-layer exfiltration path, not a TCP-layer path.** The
TCP flow metadata (`on_tcp_flow`) cannot detect it because the data
never travels over a TCP connection to the attacker. Only `on_dns_query`
sees it.

**Detection signals available in `DnsQueryContext`:**

| Signal | Field | Threshold | False positive risk |
|--------|-------|-----------|---------------------|
| Long subdomain labels | `label_lengths` | Any label > 30 chars | Low — legitimate subdomains rarely this long |
| Many labels (deep nesting) | `label_lengths.len()` | > 5 levels | Medium — some CDNs use deep subdomains |
| High total name length | `name_wire_len` | > 200 bytes | Low — DNS max is 253, normal queries ~30-60 |
| High entropy in labels | `domain` (compute Shannon entropy) | > 3.5 bits/char | Medium — CDN hashes look similar |
| High query rate to one base domain | Stateful: track per base domain | > 50 unique subdomains / 60s | Low — strong signal |
| Non-alphanumeric density | `domain` | Base64 chars (`+`, `/`, `=`) in labels | Low |

**Example policy: combined heuristic + rate detection**

```rust
struct DnsExfilDetector {
    // base_domain → (unique_subdomain_count, window_start)
    query_counts: RwLock<HashMap<String, (u64, Instant)>>,
}

impl EgressPolicy for DnsExfilDetector {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        // Heuristic: long label with high entropy
        let has_suspicious_label = ctx.label_lengths.iter()
            .zip(ctx.domain.split('.'))
            .any(|(len, label)| *len > 30 && shannon_entropy(label) > 3.5);

        if has_suspicious_label {
            return PolicyAction::Deny {
                errno: libc::EHOSTUNREACH,
                reason: PolicyReason::Custom(
                    format!("DNS exfil pattern: long high-entropy label in {}",
                            ctx.domain).into()
                ),
            };
        }

        // Rate-based: many unique subdomains to one base domain
        let base = extract_base_domain(&ctx.domain);
        let mut counts = self.query_counts.write().unwrap();
        let entry = counts.entry(base.clone())
            .or_insert((0, Instant::now()));

        // Reset window after 60s
        if entry.1.elapsed() > Duration::from_secs(60) {
            *entry = (0, Instant::now());
        }
        entry.0 += 1;

        if entry.0 > 50 {
            return PolicyAction::Deny {
                errno: libc::EHOSTUNREACH,
                reason: PolicyReason::Custom(
                    format!("DNS exfil rate: {} queries to {} in 60s",
                            entry.0, base).into()
                ),
            };
        }

        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}
```

### Entropy Analysis for Detection

Shannon entropy measures the randomness of a string. Natural language
hostnames (`archive.ubuntu.com`, `api.github.com`) have low entropy
because they use common letter patterns. Encoded data (base64, hex,
compressed) has high entropy because the byte distribution is near-uniform.

**Entropy scale for DNS labels:**

| Entropy (bits/char) | Interpretation | Examples |
|---------------------|----------------|----------|
| 1.0 – 2.5 | Natural language | `archive`, `ubuntu`, `github` |
| 2.5 – 3.5 | Mixed / ambiguous | `cdn-a1b2c3`, `edge-us-west` |
| 3.5 – 4.5 | Likely encoded data | `aGVsbG8gd29ybGQ` (base64) |
| 4.5 – 5.0 | Near-random / compressed | `7f3a9c2e` (hex), encrypted blobs |

The threshold of ~3.5 bits/char separates most legitimate hostnames from
encoded payloads. Combined with label length (>30 chars), this produces
a low false positive rate — short high-entropy labels are common (CDN
hashes like `d1234.cloudfront.net`), but long high-entropy labels are
rare in legitimate traffic.

**Implementation in `motlie_vnet::policy::entropy`:**

Network detection primitives live in `libs/vnet/src/policy/` — these
are network-specific utilities scoped to the vnet crate.

```
libs/vnet/src/policy/
├── mod.rs              // pub mod entropy; pub mod domain;
├── entropy.rs          // shannon_entropy()
├── domain.rs           // extract_base_domain(), matches_suffix()
└── (future: ratio.rs, duration.rs, beacon.rs, first_seen.rs)
```

```rust
// libs/vnet/src/policy/entropy.rs

/// Shannon entropy of a byte string (bits per character).
/// Returns 0.0 for empty input.
pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    for &b in s.as_bytes() {
        freq[b as usize] += 1;
    }
    let len = s.len() as f64;
    freq.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

// libs/vnet/src/policy/domain.rs

/// Extract the base domain (last two labels) from an FQDN.
/// "a.b.c.attacker.com" → "attacker.com"
pub fn extract_base_domain(domain: &str) -> String {
    let labels: Vec<&str> = domain.split('.').collect();
    if labels.len() >= 2 {
        labels[labels.len() - 2..].join(".")
    } else {
        domain.to_string()
    }
}
```

Policy implementations use these as `motlie_vnet::policy::entropy::shannon_entropy()`
and `motlie_vnet::policy::domain::extract_base_domain()`. No separate crate needed.

**Composing entropy analysis into a policy chain:**

The `DnsExfilDetector` shown above uses `shannon_entropy` internally.
It can be chained with other policies — entropy analysis happens only
in the DNS query callback, not on every packet:

```rust
let policy = LoggingPolicy                    // OTel-compatible tracing
    .chain(CategoryPolicy::default())         // allow pkg mgrs, deny adtech
    .chain(DnsExfilDetector::new(             // entropy + rate detection
        EntropyConfig {
            min_label_len: 30,                // only check labels > 30 chars
            entropy_threshold: 3.5,           // bits/char
            rate_window: Duration::from_secs(60),
            rate_limit: 50,                   // unique subdomains per base domain
        },
    ));
```

**Tuning considerations:**

- **`entropy_threshold`**: 3.5 is conservative. Raising to 4.0 reduces
  false positives (CDN hashes) but may miss smaller exfil payloads.
- **`min_label_len`**: 30 filters out normal short subdomains. Lowering
  catches smaller chunks but increases false positives on CDN patterns.
- **`rate_limit`**: 50 queries/60s per base domain is well above normal
  browsing patterns but below a sustained exfil stream. Adjust based
  on the guest workload (a build system doing many package fetches may
  legitimately query many subdomains of `*.ubuntu.com`).
- **Allowlist integration**: `CategoryPolicy` runs before
  `DnsExfilDetector` in the chain. Known package manager domains
  (which may have high-entropy CDN subdomains) return `Allow` and
  short-circuit — the exfil detector never sees them.

**Limitations:**

- Entropy analysis cannot detect low-entropy exfiltration (e.g. English
  words used as a covert channel). This is rare in practice because
  it is extremely bandwidth-inefficient.
- Very short exfil chunks (< 10 chars per label) may fall below both
  the length and entropy thresholds. Rate-based detection catches the
  aggregate pattern even when individual queries look benign.
- DNS-over-HTTPS (DoH) bypasses this entirely because queries go over
  an encrypted HTTPS connection to a DoH resolver. However, the TCP
  flow observer would see the connection to the DoH resolver, and the
  `CategoryPolicy` can flag unknown resolver IPs.

**Why DNS exfiltration is a first-class concern for motlie:**

Development VMs have access to credentials (`/.ssh/`, `/.env`, API keys)
injected via the vfs overlay. DNS exfiltration is one of the few channels
that bypasses TCP-level controls entirely — a compromised tool or
dependency in the guest can silently encode and exfiltrate secrets
through DNS queries that look like normal name resolution traffic.

The combination of:
1. `CredentialGuard` (vfs policy — blocks unauthorized file access)
2. `DnsExfilDetector` (vnet policy — catches encoded DNS queries)
3. TCP flow observation (catches large transfers to unknown hosts)

provides defense-in-depth across both stacks without requiring DPI or
TLS interception.

### SNI Extraction

The TLS ClientHello is the first data the client sends on a TLS
connection. It is sent in plaintext (before encryption is negotiated)
and contains the Server Name Indication extension — the hostname the
client wants to connect to.

The interceptor parses the first TX data packet on port 443 connections:
1. Check for TLS record header (content type 0x16, version)
2. Check for Handshake type ClientHello (0x01)
3. Skip to extensions
4. Find SNI extension (type 0x0000)
5. Extract the hostname string

This is a one-time parse per connection — subsequent data packets on
the same flow are not parsed for SNI.

### Design Constraint: No DPI

The flow callback is explicitly metadata-only. We do not:
- Buffer or inspect encrypted payload bytes
- Attempt TLS decryption
- Parse application-layer protocols inside encrypted streams
- Expose raw packet bytes to policy callbacks

This is a deliberate design constraint. Payload inspection would require
TLS MITM (CA trust, certificate pinning breakage) and violate the
zero-capability, embeddable-library design goal.

## Network Metadata Gaps

The TX/RX interceptor parses Ethernet → IP → UDP/TCP headers but
currently extracts only destination IP, port, DNS domain, and TLS SNI.
Additional fields are available in the parsed headers and in the
architecture. Each was evaluated against motlie's threat model: a
single-guest Linux VM with overlay-injected credentials.

### Gaps Worth Filling

| Field | Header location | Security value | Context type |
|-------|----------------|---------------|-------------|
| **DNS query count** (QDCOUNT) | DNS header bytes 4-5 | Legitimate stub resolvers send exactly 1 query/packet. Batching (>1) or zero queries indicates custom tooling (exfil tools, tunneling utilities). Strong supplemental signal alongside entropy and rate detection. | `DnsQueryContext.query_count: u16` |
| **DNS additional/authority counts** (ARCOUNT, NSCOUNT) | DNS header bytes 8-11 | Exfil tools encode data in TXT/NULL records in additional or authority sections. A stub resolver query should have ARCOUNT=0, NSCOUNT=0 — any non-zero value from the guest is anomalous and a strong exfil signal. | `DnsQueryContext.additional_count: u16`, `authority_count: u16` |
| **Guest PID** | Not in packets — requires guest agent | The single highest-value gap. Distinguishes `apt` (PID 1234) from `unknown-binary` (PID 9999). Enables true intent-based policy. Blocked on v1.3 vsock agent infrastructure. | `TcpConnectContext.guest_process: Option<String>` (already defined) |

**Implementation effort:** DNS counts are ~5 LOC each (fixed header
offsets, parsed during existing DNS extraction). Guest PID requires
v1.3 infrastructure (see below).

### Gaps Evaluated and Rejected

| Field | Header location | Why not useful for motlie |
|-------|----------------|--------------------------|
| ~~**IP TTL**~~ | IPv4 byte 8 | Guest is always one hop from libslirp. TTL is always the guest OS default (64 for Linux). Never varies between processes, never indicates tunneling in this architecture. TTL fingerprinting detects remote hosts behind proxies — irrelevant when the guest is local. |
| ~~**TCP initial window size**~~ | TCP SYN bytes 14-15 | Guest is always the same Linux kernel. Window size is a kernel TCP parameter, identical for all guest processes. `apt` and `malware` have the same TCP window. OS fingerprinting distinguishes different hosts/OSes — we have one guest OS. |
| ~~**TCP options**~~ (MSS, SACK, timestamps) | TCP SYN options | Same kernel → same TCP stack → same options for all processes. Custom userspace TCP stacks (that would produce different options) are vanishingly rare in development VMs. The theoretical detection value does not justify the parsing complexity. |
| ~~**Ethernet source MAC**~~ | Ethernet bytes 6-11 | We configure the MAC in VnetConfig. MAC spoofing detection matters on shared LANs with multiple hosts. Our virtual network has one guest and one gateway — there is no other host to deceive by spoofing. |
| ~~**libslirp connection_info()**~~ | `slirp_connection_info()` FFI | Returns a human-readable string (not a struct). Provides no information the packet-level interceptor doesn't already have. Fragile to parse across libslirp versions. Debugging utility, not a detection signal. |
| ~~**libslirp fd→connection mapping**~~ | `register_poll_fd()` callbacks | Callback provides raw fd only, no connection metadata. Even if we mapped fds to connections, it adds nothing beyond what the packet parser already extracts. Internal libslirp state with no policy value. |

### Guest Process Identity: Why vnet Cannot Do What vfs Can

The vfs and vnet stacks both want process identity (which PID initiated
this operation?), but the gaps are **fundamentally different** because
the transport architectures are different:

**vfs transport: vsock with motlie's own wire protocol**

```
Guest process → FUSE syscall → kernel FUSE driver → fuse_in_header
    (PID is here ──────────────────────────────────────────┘)
        │
        ▼
    FuseClient (guest, our code) → vsock → FsOp → host FsServer
```

The kernel FUSE driver stamps every request with the caller's PID, UID,
and GID in `fuse_in_header`. This is part of the FUSE protocol — the
kernel gives it to us for free. Our `FuseClient` receives a `Request`
object with `req.pid()` → `u32`. The gap is purely plumbing: the guest
client has the data but doesn't forward it. Fix: ~95 lines, 4 files,
no guest-side infrastructure needed.

**vnet transport: standard Ethernet over virtio-net**

```
Guest process → socket() + connect() → kernel TCP/IP stack → NIC driver
    (PID is here ─────┘)                         │
                                    packet on the wire (PID is GONE)
                                                  │
                                                  ▼
                              virtio-net → vhost-user → our interceptor
```

The kernel TCP/IP stack **strips all process identity** when it
constructs the Ethernet frame. A TCP SYN from `apt` (PID 1234) is
byte-identical to a TCP SYN from `malware` (PID 9999) — both are just
Ethernet/IP/TCP headers with source port, destination port, and flags.
PID, UID, GID, process name — none of this is in the packet. This is
not a motlie limitation; it is how TCP/IP works. No amount of plumbing
on the vnet side can recover it.

**Why vsock doesn't help for vnet**

The vfs stack uses vsock as its transport — motlie controls the wire
format and can add any fields it wants (like `CallerIdentity`). The
vnet stack uses **standard Ethernet** over virtio-net — the wire format
is the TCP/IP protocol, which we cannot modify. The guest kernel's
virtio-net driver sends standard Ethernet frames, and our vhost-user
backend receives them. There is no vsock in the vnet data path.

**The only path: out-of-band guest agent**

To get PID for network connections, the guest must report it through a
**separate channel** (not through the network packets themselves):

```
Guest agent (runs inside VM, communicates over vsock)
    │
    ├─ Reads /proc/net/tcp → maps (local_port, remote_ip:port) → inode
    ├─ Reads /proc/[pid]/fd/ → maps inode → PID
    ├─ Reads /proc/[pid]/comm → maps PID → process name
    │
    ▼
Reports to host via vsock: { source_port: 48832, pid: 1234, name: "apt" }
```

The host interceptor receives these reports and maintains a lookup table.
When a TCP SYN arrives with `source_port: 48832`, the interceptor looks
up PID 1234 / "apt" and enriches `TcpConnectContext.guest_process`.

This requires:
1. A guest agent binary running in the VM (part of v1.3 russhd work)
2. A vsock control channel between agent and host (already exists for vfs)
3. Agent-side `/proc` scanning (race-prone: short-lived connections
   may close before the agent scans; mitigation: agent uses netlink
   `SOCK_DIAG` for real-time socket events)
4. Host-side lookup table with TTL (source ports are reused)

**Comparison:**

| | vfs | vnet |
|---|---|---|
| **Transport** | vsock (motlie controls wire format) | virtio-net (standard Ethernet, immutable) |
| **PID in protocol?** | Yes — FUSE `fuse_in_header.pid` | No — TCP/IP has no PID field |
| **Fix complexity** | Plumbing: forward existing data (~95 LOC) | Infrastructure: guest agent + vsock channel + /proc scanning |
| **Dependency** | None (data already available) | v1.3 (russhd/vsock agent) |
| **Race conditions** | None (FUSE request is synchronous) | Yes (short-lived connections may close before agent scans) |

The `guest_process: Option<String>` field is already in
`TcpConnectContext` and `TcpFlowContext`, waiting for this integration.

### Priority Assessment (actionable gaps only)

| Gap | Detection value | Effort | Priority |
|-----|----------------|--------|----------|
| DNS QDCOUNT | High — detects batching/malformed queries | ~5 LOC (fixed offset) | P1: add during Phase 1 packet parser |
| DNS ARCOUNT/NSCOUNT | High — detects data in extra RR sections | ~5 LOC (fixed offset) | P1: add during Phase 1 packet parser |
| Guest PID | Highest — enables intent-based policy | High (v1.3 agent infrastructure) | P1 priority, blocked on v1.3 |

The DNS count fields should be added when the packet parser is
implemented (Phase 1 of the policy PLAN) — they are at fixed header
offsets in the DNS message and add near-zero parsing cost. Guest PID
is the highest-value gap but requires the v1.3 vsock agent
infrastructure (see section below).

## Policy Event Emission

Event emission is **infrastructure at the call site** (the slirp thread
loop), not a policy concern. Policies are pure decision functions — they
don't know their decisions are being observed.

### PolicyEvent Enum

```rust
/// Emitted by the slirp thread after every policy evaluation.
/// Identifies which callback was invoked and carries the context +
/// composite action (with confidence).
pub enum PolicyEvent {
    DnsQuery {
        ctx: DnsQueryContext,
        action: PolicyAction,
    },
    DnsResponse {
        ctx: DnsResponseContext,
        action: PolicyAction,
    },
    TcpConnect {
        ctx: TcpConnectContext,
        action: PolicyAction,
    },
    TcpFlow {
        ctx: TcpFlowContext,
        action: PolicyAction,
    },
}
```

### Emission at the Call Site

The slirp thread loop already parses frames, calls the policy chain,
and enforces the decision. Event emission is a `try_send` added between
the decision and enforcement steps:

```rust
// In slirp_thread_main — NOT inside a policy:
let ctx = parse_dns_query(&frame);
let action = policy.on_dns_query(&ctx);           // pure decision

if let Some(ref tx) = event_tx {                   // infrastructure
    let _ = tx.try_send(PolicyEvent::DnsQuery {
        ctx: ctx.clone(),
        action: action.clone(),
    });
}

match action {                                      // enforcement
    PolicyAction::Deny { errno, .. } => inject_nxdomain(&frame, errno),
    _ => slirp.input(&frame),
}
```

If no event subscriber is configured, `event_tx` is `None` and the
branch is skipped — zero overhead.

### Cross-Stack Correlation and Correlator Design

The correlator is a **host-runtime component** (not part of motlie-vnet
or motlie-vfs). It receives policy events from both stacks via a unified
`mpsc` channel, accumulates state over time, and writes back to
`InterceptPolicy` blocklists via `ArcSwap` when confidence thresholds
are crossed.

#### Assumptions and Preconditions

| Precondition | Status | Impact if not met |
|---|---|---|
| **vfs emits PolicyEvent::FsOp with path, is_sensitive** | Ready (dispatch layer enrichment) | Correlator cannot see filesystem activity |
| **vnet emits PolicyEvent::DnsQuery with domain, confidence** | Ready (PolicyEvaluator) | Correlator cannot see DNS activity |
| **vnet emits PolicyEvent::TcpFlow with bytes_tx/rx, SNI** | Ready (PolicyEvaluator) | Correlator cannot see flow patterns |
| **vfs caller_pid available** | **NOT YET** (FUSE plumbing gap, ~95 LOC) | Cannot correlate "which process read the file" with "which process opened the socket" |
| **vnet guest_process available** | **NOT YET** (v1.3 vsock agent) | Cannot attribute network connections to specific guest processes |
| **Cross-stack PID correlation** | **NOT YET** (requires both above) | Cannot detect "PID 1234 read /.env then PID 1234 connected to pastebin.com" |

Without PID, the correlator can still detect patterns based on
**timing proximity** ("a sensitive file was read, then 2 seconds later
a DNS query for an unknown domain occurred") but cannot prove the
same process did both. PID plumbing (vfs: ~95 LOC, vnet: v1.3 agent)
is the highest-value gap for cross-stack correlation.

#### Full Construction Example

```rust
use std::sync::Arc;
use std::collections::HashSet;
use std::net::Ipv4Addr;
use std::time::Duration;
use arc_swap::ArcSwap;

use motlie_vfs::{FsServer, FsInterceptPolicy, LoggingPolicy as FsLoggingPolicy, ScanDetector};
use motlie_vnet::{
    VnetConfig, VnetBackend, InterceptPolicy, LoggingPolicy as NetLoggingPolicy,
    CategoryPolicy, DnsExfilDetector, EntropyConfig,
};

fn main() -> anyhow::Result<()> {
    // ---------------------------------------------------------------
    // Step 1: Create shared blocklists (correlator writes, policies read)
    // ---------------------------------------------------------------
    let fs_blocked: Arc<ArcSwap<HashSet<String>>> =
        Arc::new(ArcSwap::from_pointee(HashSet::new()));
    let net_blocked: Arc<ArcSwap<HashSet<(Ipv4Addr, u16)>>> =
        Arc::new(ArcSwap::from_pointee(HashSet::new()));

    // ---------------------------------------------------------------
    // Step 2: Create unified event channel
    // ---------------------------------------------------------------
    enum RuntimeEvent {
        Fs(motlie_vfs::PolicyEvent),
        Net(motlie_vnet::PolicyEvent),
    }

    let (tx, rx) = std::sync::mpsc::channel::<RuntimeEvent>();
    let tx_fs = tx.clone();
    let tx_net = tx.clone();

    // ---------------------------------------------------------------
    // Step 3: Build vfs stack with PolicyEvaluator
    // ---------------------------------------------------------------
    let fs_server = FsServer::builder()
        .policy(
            FsLoggingPolicy                                 // logs every fs op via OTel tracing
                .chain(FsInterceptPolicy::new(fs_blocked.clone()))  // reads correlator blocklist
                .chain(ScanDetector)                        // flags rapid metadata scanning
        )
        .deny_threshold(0.5)                                // Allow < 0.5 → auto Deny
        .policy_events(tx_fs)                               // events → correlator
        .build();

    // Add mounts (existing FsServer API)
    fs_server.add_mount("alice-home", "/home/dchung".into(), false)?;

    // ---------------------------------------------------------------
    // Step 4: Build vnet stack with PolicyEvaluator
    // ---------------------------------------------------------------
    let vnet_config = VnetConfig::builder()
        .socket_path("/tmp/motlie-vnet-0.sock")
        .egress_policy(
            NetLoggingPolicy                                // logs every DNS/TCP via OTel tracing
                .chain(InterceptPolicy::new(net_blocked.clone()))   // reads correlator blocklist
                .chain(CategoryPolicy::default())           // classifies domains by category
                .chain(DnsExfilDetector::new(EntropyConfig {
                    min_label_len: 30,
                    entropy_threshold: 3.5,
                    rate_window: Duration::from_secs(60),
                    rate_limit: 50,
                }))
        )
        .deny_threshold(0.5)                                // Allow < 0.5 → auto Deny
        .policy_events(tx_net)                              // events → correlator
        .build()?;

    let vnet_handle = VnetBackend::new(vnet_config).start()?;
    eprintln!("vnet: socket ready at /tmp/motlie-vnet-0.sock");

    // ---------------------------------------------------------------
    // Step 5: Start correlator thread
    // ---------------------------------------------------------------
    let fs_blocked_for_correlator = fs_blocked.clone();
    let net_blocked_for_correlator = net_blocked.clone();

    std::thread::Builder::new()
        .name("correlator".into())
        .spawn(move || {
            correlator_main(rx, fs_blocked_for_correlator, net_blocked_for_correlator);
        })?;

    // ---------------------------------------------------------------
    // Step 6: Launch CH guest (existing workflow)
    // ---------------------------------------------------------------
    eprintln!("Launch CH: ./launch-ch.sh --admin-net=tap --egress-net=vhost-user");

    // ... wait for shutdown signal ...
    Ok(())
}
```

#### Correlator with Concrete Event Handling

```rust
fn correlator_main(
    rx: mpsc::Receiver<RuntimeEvent>,
    fs_blocked: Arc<ArcSwap<HashSet<String>>>,
    net_blocked: Arc<ArcSwap<HashSet<(Ipv4Addr, u16)>>>,
) {
    struct PendingRead {
        path: String,
        confidence: f64,
        timestamp: Instant,
    }

    struct TrackedFlow {
        domain: Option<String>,
        dns_confidence: f64,
        bytes_tx: u64,
        started: Instant,
    }

    let mut pending_reads: Vec<PendingRead> = Vec::new();
    let mut tracked_flows: HashMap<(Ipv4Addr, u16), TrackedFlow> = HashMap::new();

    for event in rx {
        // Expire old state (sliding 60s window)
        pending_reads.retain(|r| r.timestamp.elapsed() < Duration::from_secs(60));

        match event {
            // -------------------------------------------------------
            // vfs: sensitive file accessed
            // -------------------------------------------------------
            RuntimeEvent::Fs(motlie_vfs::PolicyEvent::FsOp { ref ctx, ref action }) => {
                if ctx.is_sensitive {
                    // Example event:
                    // FsOp {
                    //   ctx: { op: Read, path: "/.env", is_sensitive: true,
                    //          is_overlay: false, metadata_rate: 12 },
                    //   action: Allow { reason: Category(CredentialAccess), confidence: 0.4 }
                    // }
                    pending_reads.push(PendingRead {
                        path: ctx.path.clone(),
                        confidence: action.confidence(),
                        timestamp: Instant::now(),
                    });
                    eprintln!("[correlator] sensitive read: {} (confidence: {:.1})",
                              ctx.path, action.confidence());
                }
            }

            // -------------------------------------------------------
            // vfs: high-entropy write to temp (staging)
            // -------------------------------------------------------
            RuntimeEvent::Fs(motlie_vfs::PolicyEvent::FsComplete {
                ref ctx, ref result, ..
            }) => {
                if let Some(entropy) = result.write_entropy {
                    if entropy > 4.0 && ctx.path.starts_with("/tmp/") {
                        // Example event:
                        // FsComplete {
                        //   ctx: { op: Write, path: "/tmp/payload.bin" },
                        //   result: { bytes: Some(2_000_000), write_entropy: Some(4.8) },
                        //   action: Allow { confidence: 1.0 }
                        // }
                        pending_reads.push(PendingRead {
                            path: ctx.path.clone(),
                            confidence: 0.3,  // staging = suspicious
                            timestamp: Instant::now(),
                        });
                        eprintln!("[correlator] staging detected: {} (entropy: {:.1}, {}B)",
                                  ctx.path, entropy, result.bytes.unwrap_or(0));
                    }
                }
            }

            // -------------------------------------------------------
            // vnet: uncertain DNS allow
            // -------------------------------------------------------
            RuntimeEvent::Net(motlie_vnet::PolicyEvent::DnsQuery { ref ctx, ref action }) => {
                // Example event:
                // DnsQuery {
                //   ctx: { domain: "suspicious-cdn.xyz", query_type: A,
                //          label_lengths: [15, 3], name_wire_len: 20 },
                //   action: Allow { reason: Category(Unknown), confidence: 0.5 }
                // }
                if action.confidence() < 0.8 {
                    eprintln!("[correlator] uncertain DNS: {} (confidence: {:.1})",
                              ctx.domain, action.confidence());
                }
            }

            // -------------------------------------------------------
            // vnet: TCP connection established
            // -------------------------------------------------------
            RuntimeEvent::Net(motlie_vnet::PolicyEvent::TcpConnect { ref ctx, ref action }) => {
                // Example event:
                // TcpConnect {
                //   ctx: { dst_ip: 93.184.216.34, dst_port: 443,
                //          domain: Some("suspicious-cdn.xyz"), source_port: 48832 },
                //   action: Allow { reason: Category(Unknown), confidence: 0.5 }
                // }
                tracked_flows.insert(
                    (ctx.dst_ip, ctx.dst_port),
                    TrackedFlow {
                        domain: ctx.domain.clone(),
                        dns_confidence: action.confidence(),
                        bytes_tx: 0,
                        started: Instant::now(),
                    },
                );
            }

            // -------------------------------------------------------
            // vnet: flow update — check for exfiltration
            // -------------------------------------------------------
            RuntimeEvent::Net(motlie_vnet::PolicyEvent::TcpFlow { ref ctx, ref action }) => {
                // Example event:
                // TcpFlow {
                //   ctx: { dst_ip: 93.184.216.34, dst_port: 443,
                //          domain: Some("suspicious-cdn.xyz"),
                //          sni: Some("suspicious-cdn.xyz"),
                //          bytes_tx: 5_200_000, bytes_rx: 1_200,
                //          duration: 8s, is_close: false },
                //   action: Allow { confidence: 0.5 }
                // }
                let key = (ctx.dst_ip, ctx.dst_port);
                if let Some(flow) = tracked_flows.get_mut(&key) {
                    flow.bytes_tx = ctx.bytes_tx;

                    // Cross-stack check: sensitive read + uncertain flow + large upload
                    let has_recent_sensitive = pending_reads.iter().any(|r| {
                        r.timestamp.elapsed() < Duration::from_secs(60)
                    });

                    if has_recent_sensitive && flow.dns_confidence < 0.8 && flow.bytes_tx > 1_000_000 {
                        // Compound signal!
                        let compound_confidence = 0.9;
                        eprintln!(
                            "[correlator] EXFIL DETECTED: {}:{} domain={:?} \
                             bytes_tx={} dns_confidence={:.1} → BLOCKING",
                            ctx.dst_ip, ctx.dst_port, flow.domain,
                            flow.bytes_tx, flow.dns_confidence,
                        );

                        // Write to vnet blocklist
                        let mut current = (**net_blocked.load()).clone();
                        current.insert(key);
                        net_blocked.store(Arc::new(current));

                        // Optionally also block the staged file in vfs
                        for read in &pending_reads {
                            if read.path.starts_with("/tmp/") {
                                let mut fs_current = (**fs_blocked.load()).clone();
                                fs_current.insert(read.path.clone());
                                fs_blocked.store(Arc::new(fs_current));
                                eprintln!("[correlator] also blocking fs path: {}", read.path);
                            }
                        }
                    }
                }

                // Clean up closed flows
                if ctx.is_close {
                    tracked_flows.remove(&key);
                }
            }

            _ => {}
        }
    }
}
```

#### Example Event Stream

A concrete sequence showing the events the correlator receives, with
actual `PolicyAction` values and confidence scores:

```
T+0.0s  ← Fs(FsOp {
            ctx: { op: Read, tag: "alice-home", path: "/.env",
                   is_sensitive: true, is_overlay: false,
                   metadata_rate: 12, unique_paths_in_window: 5 },
            action: Allow { reason: Some(Category(CredentialAccess)),
                            confidence: 0.4 }
          })
        → correlator: records pending sensitive read "/.env" at 0.4

T+1.2s  ← Fs(FsComplete {
            ctx: { op: Read, path: "/.env" },
            result: { success: true, bytes: Some(256), write_entropy: None,
                      latency: 0.3ms },
            action: Allow { confidence: 1.0 }
          })
        → correlator: notes 256 bytes read (small, likely API key)

T+2.1s  ← Net(DnsQuery {
            ctx: { domain: "drop.evil-cdn.xyz",
                   query_type: A, source_port: 43210,
                   label_lengths: [4, 8, 3], name_wire_len: 20 },
            action: Allow { reason: Some(Category(Unknown)),
                            confidence: 0.5 }
          })
        → correlator: uncertain DNS 2.1s after sensitive read — tracking

T+2.3s  ← Net(TcpConnect {
            ctx: { dst_ip: 198.51.100.42, dst_port: 443,
                   domain: Some("drop.evil-cdn.xyz"),
                   source_port: 48832 },
            action: Allow { reason: Some(Category(Unknown)),
                            confidence: 0.5 }
          })
        → correlator: flow started to uncertain domain — tracking

T+3.5s  ← Net(TcpFlow {
            ctx: { dst_ip: 198.51.100.42, dst_port: 443,
                   domain: Some("drop.evil-cdn.xyz"),
                   sni: Some("drop.evil-cdn.xyz"),
                   bytes_tx: 52_000, bytes_rx: 800,
                   duration: 1.2s, is_close: false },
            action: Allow { confidence: 0.5 }
          })
        → correlator: 52KB uploaded — not yet threshold (1MB)

T+8.7s  ← Net(TcpFlow {
            ctx: { dst_ip: 198.51.100.42, dst_port: 443,
                   bytes_tx: 2_100_000, bytes_rx: 1_200,
                   duration: 6.4s, is_close: false },
            action: Allow { confidence: 0.5 }
          })
        → correlator: 2.1MB uploaded!
          sensitive read (T+0, confidence 0.4) +
          uncertain DNS (confidence 0.5) +
          large upload (2.1MB tx, 1.2KB rx = 1750:1 ratio)
          → compound confidence: 0.9
          → WRITE net_blocked: { (198.51.100.42, 443) }

T+8.8s  (next packet from guest to 198.51.100.42:443)
        → InterceptPolicy reads ArcSwap → match!
        → Deny { errno: ECONNREFUSED, reason: "correlator: blocked",
                  confidence: 1.0 }
        → forge TCP RST → inject into guest
        → connection killed, exfiltration stopped
```

#### Correlator Thread

```rust
// Correlator: separate thread, reads from unified channel,
// writes to both blocklists.

struct CorrelatorState {
    // Recent filesystem events (sliding window)
    recent_fs_events: VecDeque<(Instant, motlie_vfs::FsOpContext, PolicyAction)>,
    // Recent network events
    recent_net_events: VecDeque<(Instant, NetEventSummary)>,
    // Known sensitive file accesses awaiting network correlation
    pending_sensitive_reads: Vec<SensitiveReadRecord>,
    // Active flow tracking for exfiltration detection
    active_flows: HashMap<(Ipv4Addr, u16), FlowAccumulator>,
}

struct SensitiveReadRecord {
    path: String,
    timestamp: Instant,
    caller_pid: Option<u32>,  // None until FUSE PID plumbed
}

struct FlowAccumulator {
    domain: Option<String>,
    bytes_tx: u64,
    first_seen: Instant,
    dns_confidence: f64,  // confidence from the DNS Allow
}

thread::spawn(move || {
    let mut state = CorrelatorState::new();

    for event in rx {
        match event {
            RuntimeEvent::Fs(motlie_vfs::PolicyEvent::FsOp { ctx, action }) => {
                state.handle_fs_op(&ctx, &action, &fs_blocked, &net_blocked);
            }
            RuntimeEvent::Fs(motlie_vfs::PolicyEvent::FsComplete { ctx, result, action }) => {
                state.handle_fs_complete(&ctx, &result, &action);
            }
            RuntimeEvent::Net(motlie_vnet::PolicyEvent::DnsQuery { ctx, action }) => {
                state.handle_dns(&ctx, &action, &net_blocked);
            }
            RuntimeEvent::Net(motlie_vnet::PolicyEvent::TcpConnect { ctx, action }) => {
                state.handle_tcp_connect(&ctx, &action);
            }
            RuntimeEvent::Net(motlie_vnet::PolicyEvent::TcpFlow { ctx, action }) => {
                state.handle_tcp_flow(&ctx, &action, &net_blocked);
            }
            _ => {}
        }

        // Periodic: expire old events, run cross-stack correlation
        state.expire_old_events(Duration::from_secs(60));
        state.run_cross_correlation(&fs_blocked, &net_blocked);
    }
});
```

#### Cross-Stack Detection: Credential Read → DNS Exfiltration

The highest-value cross-stack pattern: a sensitive file is read via vfs,
then shortly after, a suspicious DNS query or large upload occurs via
vnet. Neither event alone is definitive — combined, they are a strong
exfiltration signal.

```
Timeline:

T+0s    vfs: Read /.env → Allow { confidence: 0.4, reason: "sensitive file" }
        → correlator records: SensitiveReadRecord { path: "/.env", t: T+0 }

T+2s    vnet: DNS query for suspicious-cdn.xyz → Allow { confidence: 0.5 }
        → correlator: recent sensitive read + uncertain DNS → bump suspicion

T+3s    vnet: TCP connect to 93.184.216.34:443 (resolved from suspicious-cdn.xyz)
        → correlator: tracks flow

T+10s   vnet: TcpFlow { bytes_tx: 5MB, domain: "suspicious-cdn.xyz" }
        → correlator: sensitive read within 60s + uncertain DNS +
          large upload to unknown domain
        → compound confidence: 0.9
        → ACTION: net_blocked.store(Arc::new({ (93.184.216.34, 443) }))
        → next packet to 93.184.216.34:443 → InterceptPolicy → Deny

T+10.1s vnet: next frame to 93.184.216.34:443
        → InterceptPolicy reads blocklist → match → Deny { confidence: 1.0 }
        → forge TCP RST → connection killed
```

**Correlator logic for this pattern:**

```rust
impl CorrelatorState {
    fn run_cross_correlation(
        &mut self,
        fs_blocked: &Arc<ArcSwap<HashSet<String>>>,
        net_blocked: &Arc<ArcSwap<HashSet<(Ipv4Addr, u16)>>>,
    ) {
        // Pattern: sensitive file read followed by large upload to unknown domain
        for flow in self.active_flows.values() {
            if flow.bytes_tx < 1_000_000 { continue; }  // <1MB not interesting
            if flow.dns_confidence > 0.8 { continue; }  // known domain

            // Check if a sensitive file was read recently
            let recent_sensitive = self.pending_sensitive_reads.iter().any(|r| {
                r.timestamp.elapsed() < Duration::from_secs(60)
            });

            if recent_sensitive {
                // Compound signal: sensitive read + low-confidence DNS + large upload
                let compound_confidence = 0.9;

                if compound_confidence > 0.8 {
                    // Block the destination
                    if let Some(domain) = &flow.domain {
                        // Also block the domain's other IPs if known
                    }
                    let mut current = (**net_blocked.load()).clone();
                    // Insert all IPs associated with this flow
                    // current.insert((flow_dst_ip, flow_dst_port));
                    net_blocked.store(Arc::new(current));
                }
            }
        }
    }
}
```

#### Cross-Stack Detection: Staging → Execution

Another pattern: a high-entropy file is written to a temp directory
(staging), then a new outbound connection is established (exfiltration
or C2 callback).

```
Timeline:

T+0s    vfs: Write /tmp/payload.bin → on_complete: write_entropy=4.8, bytes=2MB
        → correlator: staging pattern (high entropy write to temp)

T+5s    vnet: DNS query for never-seen-before.io → Allow { confidence: 0.3 }
        → correlator: staging + novel domain

T+6s    vnet: TCP connect to resolved IP:4444
        → correlator: staging + novel domain + unusual port (4444)
        → compound confidence: 0.95
        → ACTION: block IP:4444, also block path /tmp/payload.bin
          (vfs InterceptPolicy → future reads of /tmp/payload.bin denied)
```

#### ML / LLM as Correlator (Future)

The correlator is a function: `stream<RuntimeEvent> → write(blocklists)`.
Nothing constrains this to be rule-based code. Future implementations
could replace or augment the rule-based correlator with:

**Supervised ML model:** trained on labeled event streams (normal dev
workflows vs attack scenarios). Input: sliding window of `RuntimeEvent`s
with extracted features (event type, confidence, timing deltas, byte
volumes, entropy values). Output: per-destination threat score (0.0–1.0).
When score exceeds threshold → write to blocklist.

**LLM-based correlator:** the event stream is structured data that an
LLM can reason about. A prompt-based correlator could:
- Receive batches of recent events as structured JSON
- Apply natural-language reasoning ("the guest read SSH keys, then
  resolved a domain registered 2 days ago, then uploaded 5MB —
  this looks like credential exfiltration")
- Return a structured decision: `{ action: "block", target: "1.2.3.4:443",
  confidence: 0.92, reasoning: "..." }`
- The host runtime parses the response and writes to the blocklist

This is feasible because:
- The event stream is structured and bounded (not raw packet data)
- Decisions are asynchronous (correlator runs off the hot path)
- Latency tolerance is high (seconds, not nanoseconds)
- The blocklist write is the same `ArcSwap` mechanism regardless of
  whether the correlator is rules, ML, or LLM

The policy API and event pipeline do not need to change — the correlator
is a pluggable consumer of the `mpsc::Receiver<RuntimeEvent>` channel.

#### Future: Full Guest Quarantine

When the correlator reaches high compound confidence that a guest is
compromised, the current design blocks individual destinations (IP:port
in vnet, path in vfs). A future enhancement is **full guest quarantine**
— locking down both stacks simultaneously in real-time:

**Network quarantine:**
- Write a wildcard entry to `net_blocked` that the `InterceptPolicy`
  interprets as "deny all outbound" (e.g. a sentinel value like
  `(Ipv4Addr::UNSPECIFIED, 0)`)
- All subsequent DNS queries → NXDOMAIN, all TCP SYNs → RST
- The guest's network is instantly dead — no exfiltration possible
- Existing established flows: forge RST for each tracked flow in
  the correlator's `active_flows` map

**Filesystem quarantine:**
- Write a wildcard entry to `fs_blocked` that `FsInterceptPolicy`
  interprets as "deny all mutations" (e.g. sentinel path `"*"`)
- All Write/Create/Unlink/Rename/Mkdir/Rmdir → `EACCES`
- Reads remain allowed (forensic access to understand what happened)
- Alternatively: deny all operations including reads for full isolation

**Quarantine signal:**

```rust
/// Sent from correlator to both stacks simultaneously.
struct QuarantineSignal {
    guest_id: String,
    reason: String,
    compound_confidence: f64,
    /// Which stacks to lock down.
    network: bool,   // kill all outbound
    filesystem: bool, // kill all mutations (or all ops)
}
```

The correlator writes to both blocklists atomically (two `ArcSwap`
stores). The next operation on either stack hits the sentinel entry
and is denied. Latency from decision to enforcement: ~25ns (one
`ArcSwap` read).

**Recovery:** quarantine is reversible — the correlator (or a human
operator via the REPL) clears the sentinel entries from the blocklists.
The guest resumes normal operation. This enables a "quarantine, inspect,
release" workflow for false positives.

**Prerequisites:** same as the correlator itself — no new API needed.
The `InterceptPolicy` already reads from `ArcSwap`; the only change is
recognizing a sentinel value as "deny all." This is a policy
implementation detail, not an architecture change.

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Host Runtime                           │
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ FsServer         │      │ VnetBackend      │           │
│  │ + PolicyEvaluator│      │ + PolicyEvaluator│           │
│  │                  │      │                  │           │
│  │ LoggingPolicy    │      │ LoggingPolicy    │           │
│  │ .chain(Intercept)│      │ .chain(Intercept)│           │
│  │ .chain(Scan)     │      │ .chain(Category) │           │
│  │                  │      │ .chain(Exfil)    │           │
│  └───────┬──────────┘      └────────┬─────────┘           │
│          │ PolicyEvent::FsOp        │ PolicyEvent::DnsQuery│
│          │ PolicyEvent::FsComplete  │ PolicyEvent::TcpFlow │
│          │                          │                      │
│          ▼                          ▼                      │
│  ┌──────────────────────────────────────────────┐         │
│  │              mpsc::channel                    │         │
│  │         rx: Receiver<RuntimeEvent>            │         │
│  └──────────────────┬───────────────────────────┘         │
│                     │                                      │
│                     ▼                                      │
│  ┌──────────────────────────────────────────────┐         │
│  │              Correlator Thread                │         │
│  │                                               │         │
│  │  Rules / ML model / LLM                      │         │
│  │                                               │         │
│  │  Accumulates state:                           │         │
│  │  - recent sensitive file reads               │         │
│  │  - uncertain DNS allows                       │         │
│  │  - active flow byte counts                   │         │
│  │  - timing correlation                         │         │
│  │                                               │         │
│  │  When compound confidence > threshold:       │         │
│  │                                               │         │
│  │  ┌──────────────┐    ┌──────────────┐        │         │
│  │  │fs_blocked    │    │net_blocked   │        │         │
│  │  │ArcSwap<Hash> │    │ArcSwap<Hash> │        │         │
│  │  │(path blocklist)   │(IP:port list)│        │         │
│  │  └──────┬───────┘    └──────┬───────┘        │         │
│  └─────────┼───────────────────┼────────────────┘         │
│            │                   │                           │
│            │ ~25ns read        │ ~25ns read                │
│            ▼                   ▼                           │
│   FsInterceptPolicy    InterceptPolicy                    │
│   in FsServer chain    in VnetBackend chain               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Neither `motlie-vfs` nor `motlie-vnet` depends on the other. Each crate
provides: `PolicyEvaluator`, `PolicyEvent` enum, `InterceptPolicy` with
`ArcSwap`, and the `EgressPolicy`/`FsPolicy` trait. The host runtime
owns the unified channel, the correlator, and the blocklists.

### Design Alignment with motlie-vfs

Both `motlie-vfs` and `motlie-vnet` follow the same policy patterns
but are separate trait hierarchies — filesystem and network policies
do not intermingle because they operate on fundamentally different stacks:

| Aspect | vfs (`FsPolicy`) | vnet (`EgressPolicy`) |
|--------|-------------------|----------------------|
| Context | `FsOpContext` (op, tag, path, overlay) | `DnsQueryContext`, `TcpConnectContext`, `TcpFlowContext` |
| Deny mechanism | Return errno to FUSE layer | Forge NXDOMAIN/RST (mapped from errno) |
| Post-op hook | `on_complete()` with result + bytes + latency | `on_tcp_flow()` with byte counts + duration |
| Chaining | `.chain()` → `impl FsPolicy` | `.chain()` → `impl EgressPolicy` |
| Default | `NoPolicy` (compiled out) | `NoPolicy` (compiled out) |
| Action type | `PolicyAction` (same shape) | `PolicyAction` (same shape) |
| Reason type | `PolicyReason` (Category or Custom Cow) | `PolicyReason` (same shape) |
| Event sink | `EventSinkPolicy<F>` | `EventSinkPolicy<F>` |
| Thread safety | `Send + Sync` (multi-thread vsock) | `Send` (single slirp thread) |

### Future: Cross-Stack Event Correlation

The `EventSinkPolicy` in both crates enables a host-runtime correlator
that sees both filesystem and network activity:

```
FsPolicy chain ──→ EventSinkPolicy ──→ mpsc::Sender<RuntimeEvent>
                                              │
EgressPolicy chain ──→ EventSinkPolicy ──→────┘
                                              │
                                              ▼
                                       Correlator thread
                                       (sees both stacks)
```

Example correlation signals (host runtime concern, not vnet):
- "read `/.env` then DNS lookup for `pastebin.com`" → exfiltration
- "write to `/tmp/payload` then connect to unknown IP:4444" → reverse shell

The vnet DESIGN requirement: **context types must be `Clone + Send`** so
the event sink policy can transport them across thread boundaries.

## PolicyEvaluator and Plumbing

### PolicyEvaluator

The runtime bundle that sits between the handler and the policy chain.
Owns the chain, the threshold, and the event channel. The handler calls
`evaluate_dns()` / `evaluate_tcp()` / `evaluate_flow()` — everything
else (threshold, emission) is internal to the evaluator.

```rust
pub struct PolicyEvaluator<P: EgressPolicy> {
    policy: P,
    deny_threshold: f64,
    event_tx: Option<mpsc::Sender<PolicyEvent>>,
}

impl<P: EgressPolicy> PolicyEvaluator<P> {
    pub fn evaluate_dns(&self, ctx: &DnsQueryContext) -> PolicyAction {
        let action = self.policy.on_dns_query(ctx);
        let action = self.apply_threshold(action);
        self.emit(PolicyEvent::DnsQuery { ctx: ctx.clone(), action: action.clone() }, &action);
        action
    }

    pub fn evaluate_tcp(&self, ctx: &TcpConnectContext) -> PolicyAction {
        let action = self.policy.on_tcp_connect(ctx);
        let action = self.apply_threshold(action);
        self.emit(PolicyEvent::TcpConnect { ctx: ctx.clone(), action: action.clone() }, &action);
        action
    }

    pub fn evaluate_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
        let action = self.policy.on_tcp_flow(ctx);
        // No threshold on flow — connection already established.
        self.emit(PolicyEvent::TcpFlow { ctx: ctx.clone(), action: action.clone() }, &action);
        action
    }

    fn apply_threshold(&self, action: PolicyAction) -> PolicyAction {
        match action {
            PolicyAction::Allow { confidence, reason } if confidence < self.deny_threshold => {
                PolicyAction::Deny {
                    errno: libc::EHOSTUNREACH,
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

### Builder → VnetConfig → slirp thread

```rust
// Builder:
pub struct VnetConfigBuilder<P: EgressPolicy = NoPolicy> {
    socket_path: Option<PathBuf>,
    policy: P,
    deny_threshold: f64,
    event_tx: Option<mpsc::Sender<PolicyEvent>>,
    // ... other network config ...
}

impl VnetConfigBuilder<NoPolicy> {
    pub fn new() -> Self {
        Self {
            socket_path: None,
            policy: NoPolicy,
            deny_threshold: 0.0,
            event_tx: None,
        }
    }
}

impl<P: EgressPolicy> VnetConfigBuilder<P> {
    /// Set the policy chain. Changes the generic parameter.
    pub fn egress_policy<Q: EgressPolicy>(self, policy: Q) -> VnetConfigBuilder<Q> {
        VnetConfigBuilder {
            policy,
            deny_threshold: self.deny_threshold,
            event_tx: self.event_tx,
            socket_path: self.socket_path,
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

    pub fn build(self) -> Result<VnetConfig<P>, VnetError> {
        let policy_evaluator = if /* P is NoPolicy and no threshold and no events */ {
            None  // truly nothing to do
        } else {
            Some(PolicyEvaluator {
                policy: self.policy,
                deny_threshold: self.deny_threshold,
                event_tx: self.event_tx,
            })
        };

        Ok(VnetConfig {
            socket_path: self.socket_path.unwrap(),
            policy_evaluator,
            // ... other fields
        })
    }
}

// VnetConfig stores the evaluator:
pub struct VnetConfig<P: EgressPolicy = NoPolicy> {
    pub socket_path: PathBuf,
    pub policy_evaluator: Option<PolicyEvaluator<P>>,
    // ... guest_ipv4, host_ipv4, mac, etc ...
}
```

### Slirp thread integration

```rust
fn slirp_thread_main<P: EgressPolicy>(
    evaluator: Option<&PolicyEvaluator<P>>,
    tx_receiver: mpsc::Receiver<Vec<u8>>,
    rx_sender: mpsc::Sender<Vec<u8>>,
    slirp: &SlirpInstance,
    // ... shutdown, rx_eventfd ...
) {
    while let Ok(frame) = tx_receiver.try_recv() {
        if let Some(eval) = evaluator {
            match parse_frame(&frame) {
                ParsedPacket::DnsQuery(ctx) => {
                    let action = eval.evaluate_dns(&ctx);
                    match action {
                        PolicyAction::Deny { errno, .. } => {
                            let resp = forge_nxdomain(&frame, errno);
                            let _ = rx_sender.send(resp);
                        }
                        PolicyAction::Allow { .. } => slirp.input(&frame),
                    }
                }
                ParsedPacket::TcpSyn(ctx) => {
                    let action = eval.evaluate_tcp(&ctx);
                    match action {
                        PolicyAction::Deny { .. } => {
                            let resp = forge_tcp_rst(&frame);
                            let _ = rx_sender.send(resp);
                        }
                        PolicyAction::Allow { .. } => slirp.input(&frame),
                    }
                }
                ParsedPacket::Other => slirp.input(&frame),
            }
        } else {
            slirp.input(&frame);  // no evaluator — zero overhead
        }
    }
}
```

When `policy_evaluator` is `None`, the `if let Some` branch is a single
null-pointer check (~1ns). The parse, threshold, and emission code never
executes.

## Chaining Example: LoggingPolicy + CategoryPolicy

### Code

```rust
// --- LoggingPolicy: always Allow, logs via OTel tracing ---
pub struct LoggingPolicy;

impl EgressPolicy for LoggingPolicy {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "CLIENT",
            otel.name = "vnet.dns.query",
            net.host.name = %ctx.domain,
            net.protocol.name = "dns",
            dns.query_type = ?ctx.query_type,
            dns.label_count = ctx.label_lengths.len(),
            dns.name_wire_len = ctx.name_wire_len,
            "dns.query"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "CLIENT",
            otel.name = "vnet.tcp.connect",
            net.peer.ip = %ctx.dst_ip,
            net.peer.port = ctx.dst_port,
            net.host.name = ?ctx.domain,
            "tcp.connect"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }

    fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
        tracing::info!(
            otel.kind = "CLIENT",
            otel.name = "vnet.tcp.flow",
            net.peer.ip = %ctx.dst_ip,
            net.peer.port = ctx.dst_port,
            net.host.name = ?ctx.domain,
            net.tls.sni = ?ctx.sni,
            net.bytes.tx = ctx.bytes_tx,
            net.bytes.rx = ctx.bytes_rx,
            net.duration_ms = ctx.duration.as_millis() as u64,
            "tcp.flow"
        );
        PolicyAction::Allow { reason: None, confidence: 1.0 }
    }
}

// --- CategoryPolicy: classifies domains, decides Allow/Deny ---
pub struct CategoryPolicy { rules: Vec<CategoryRule> }

impl EgressPolicy for CategoryPolicy {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        match self.categorize(&ctx.domain) {
            DomainCategory::PackageManager => PolicyAction::Allow {
                reason: Some(PolicyReason::Category(DomainCategory::PackageManager)),
                confidence: 1.0,
            },
            DomainCategory::AdNetwork => PolicyAction::Deny {
                errno: libc::EHOSTUNREACH,
                reason: PolicyReason::Category(DomainCategory::AdNetwork),
                confidence: 1.0,
            },
            DomainCategory::Unknown => PolicyAction::Allow {
                reason: Some(PolicyReason::Category(DomainCategory::Unknown)),
                confidence: 0.5,
            },
        }
    }
    // on_tcp_connect, on_tcp_flow: similar pattern
}

// --- Chain and configure ---
let config = VnetConfig::builder()
    .socket_path("/tmp/motlie-vnet-0.sock")
    .egress_policy(
        LoggingPolicy                          // logs every operation
            .chain(CategoryPolicy::default())  // classifies and decides
    )
    .deny_threshold(0.5)
    .build()?;
```

**Note:** `LoggingPolicy` goes first because it always returns
`Allow { confidence: 1.0 }`. If `CategoryPolicy` were first and returned
`Deny`, `LoggingPolicy` would never run (Deny short-circuits). Putting
the logger first ensures every operation is traced regardless of outcome.

### Walkthrough: DNS query for `doubleclick.net` (known-bad)

```
1. LoggingPolicy.on_dns_query(ctx)
   → tracing::info!(net.host.name = "doubleclick.net", ...)
   → returns Allow { confidence: 1.0 }

2. CategoryPolicy.on_dns_query(ctx)
   → categorize("doubleclick.net") = AdNetwork
   → returns Deny { errno: EHOSTUNREACH, reason: Category(AdNetwork), confidence: 1.0 }

3. Chain merges: first Deny wins
   → composite: Deny { errno: EHOSTUNREACH, reason: Category(AdNetwork), confidence: 1.0 }

4. PolicyEvaluator.apply_threshold: Deny passes through unchanged

5. PolicyEvaluator.emit: Deny → always emitted to correlator
   → PolicyEvent::DnsQuery { ctx, action: Deny { AdNetwork, 1.0 } }

6. Slirp thread: forge NXDOMAIN response → inject into guest rx path
   Guest sees: DNS lookup for doubleclick.net → NXDOMAIN
```

### Walkthrough: DNS query for `suspicious-cdn.xyz` (unknown)

```
1. LoggingPolicy.on_dns_query(ctx)
   → tracing::info!(net.host.name = "suspicious-cdn.xyz", ...)
   → returns Allow { confidence: 1.0 }

2. CategoryPolicy.on_dns_query(ctx)
   → categorize("suspicious-cdn.xyz") = Unknown
   → returns Allow { reason: Category(Unknown), confidence: 0.5 }

3. Chain merges: both Allow, take min confidence
   → LoggingPolicy had 1.0, CategoryPolicy had 0.5
   → composite: Allow { reason: Category(Unknown), confidence: 0.5 }
   → reason from CategoryPolicy (it had the lower confidence = weakest link)

4. PolicyEvaluator.apply_threshold(0.5):
   → confidence 0.5 < threshold 0.5? → 0.5 < 0.5 is FALSE (strict less-than)
   → Allow passes through unchanged
   (if threshold were 0.6: this would become Deny { errno: EHOSTUNREACH,
    reason: Category(Unknown), confidence: 0.5 })

5. PolicyEvaluator.emit: Allow with confidence 0.5 < 1.0 → emitted
   → PolicyEvent::DnsQuery { ctx, action: Allow { Category(Unknown), 0.5 } }
   → correlator receives: "suspicious-cdn.xyz allowed at 0.5 confidence
     because CategoryPolicy classified it as Unknown"

6. Slirp thread: slirp.input(&frame) — query proceeds to host resolver
   Guest sees: DNS lookup for suspicious-cdn.xyz → resolves normally
```

### Walkthrough: DNS query for `archive.ubuntu.com` (known-good)

```
1. LoggingPolicy.on_dns_query(ctx)
   → tracing::info!(net.host.name = "archive.ubuntu.com", ...)
   → returns Allow { confidence: 1.0 }

2. CategoryPolicy.on_dns_query(ctx)
   → categorize("archive.ubuntu.com") = PackageManager
   → returns Allow { reason: Category(PackageManager), confidence: 1.0 }

3. Chain merges: both Allow { 1.0 }, min = 1.0
   → composite: Allow { reason: Category(PackageManager), confidence: 1.0 }

4. PolicyEvaluator.apply_threshold: Allow 1.0 >= any threshold → passes through

5. PolicyEvaluator.emit: Allow with confidence 1.0 → NOT emitted (not interesting)

6. Slirp thread: slirp.input(&frame) — query proceeds
   Guest sees: DNS lookup for archive.ubuntu.com → resolves normally
   Correlator sees: nothing (confident Allow skipped)
```

## Domain→IP Reverse Map

DNS interception populates a reverse map so TCP connection decisions
have domain context:

```rust
struct DomainInfo {
    domain: String,
    resolved_at: Instant,
    ttl: u32,
}

struct DnsReverseMap {
    map: HashMap<Ipv4Addr, DomainInfo>,
}

impl DnsReverseMap {
    fn insert(&mut self, ip: Ipv4Addr, domain: String, ttl: u32);
    fn lookup(&self, ip: &Ipv4Addr) -> Option<&str>;
}
```

The reverse map is owned by the interceptor on the slirp thread — no
synchronization needed.

## Packet Parsing

The interceptor needs to parse:
1. **Ethernet header** (14 bytes): ethertype → IPv4
2. **IPv4 header** (20+ bytes): protocol → UDP/TCP, src/dst IP
3. **UDP header** (8 bytes): src/dst port → port 53 = DNS
4. **TCP header** (20+ bytes): flags → SYN detection, src/dst port
5. **DNS message**: query name, type, response code, answer records

### Approach

Implement a lightweight `PacketParser` that extracts only the fields
needed for policy decisions. No full protocol stack — just header
offsets and DNS name decompression.

```rust
pub enum ParsedPacket {
    DnsQuery { domain: String, query_type: u16, src_port: u16 },
    DnsResponse { domain: String, answers: Vec<DnsAnswer>, src_port: u16 },
    TcpSyn { dst_ip: Ipv4Addr, dst_port: u16, src_port: u16 },
    Other,
}
```

### Response Forging

When policy denies a request, the interceptor forges a guest-visible
response:

- **DNS deny**: construct an NXDOMAIN response with the original query's
  transaction ID. Inject into the rx path (bypassing libslirp).
- **TCP deny**: construct a TCP RST packet with correct sequence numbers.
  Inject into the rx path.

Both require constructing valid Ethernet + IP + protocol headers with
correct checksums.

## Performance

### Per-Packet Overhead by Policy Configuration

| Configuration | Parse cost | Policy cost | Forge cost | Total |
|--------------|-----------|------------|-----------|-------|
| `NoPolicy` (default) | 0 | 0 | 0 | **0** (compiled out) |
| `LoggingPolicy` only | ~50ns (IP header) / ~300ns (DNS) | ~20ns (tracing) | 0 | ~70-320ns |
| `LoggingPolicy.chain(CategoryPolicy)` | same | ~70ns (HashMap lookup) | 0 (allow) / ~500ns (deny) | ~120-870ns |
| Any chain with deny | same | same | ~500ns (per denied packet) | ~120-870ns |

At 1000 packets/s (heavy development workload), worst-case overhead is
~1ms/s — negligible compared to libslirp's ~1-10us per packet and
syscall costs.

### Zero-Cost NoPolicy Guarantee

When `P = NoPolicy`:
- `on_dns_query()` is `{ PolicyAction::Allow { reason: None, confidence: 1.0 } }`
- The compiler inlines this → the interceptor's `if let Deny = ...` branch
  is statically false → the parser call, reverse map lookup, and forge
  call are all dead code → eliminated
- The slirp thread loop compiles to:
  ```rust
  while let Ok(frame) = tx_receiver.try_recv() {
      slirp.input(&frame);
  }
  ```
  Identical to the current no-policy code. Verified by inspecting the
  generated assembly (future: add a `#[test]` that checks binary size
  delta between NoPolicy and no-policy-support builds).

## Integration Examples

### Logging-only (no enforcement, threshold=0.0)

```rust
let config = VnetConfig::builder()
    .socket_path("/tmp/motlie-vnet-0.sock")
    .egress_policy(LoggingPolicy::new())
    .build()?;
```

### Full enforcement with chaining

```rust
let config = VnetConfig::builder()
    .socket_path("/tmp/motlie-vnet-0.sock")
    .egress_policy(
        LoggingPolicy::new()
            .chain(CategoryPolicy::from_rules(my_rules))
            .chain(my_runtime_policy)
    )
    .build()?;
```

### No policy (zero overhead, current behavior)

```rust
let config = VnetConfig::builder()
    .socket_path("/tmp/motlie-vnet-0.sock")
    .build()?;
// P = NoPolicy, interceptor compiled out
```

## Category Policy

### Domain Classification

The `CategoryPolicy` classifies domains by purpose using a configurable
rule set. The library ships a default rule set but the host runtime can
override it.

```rust
pub enum DomainCategory {
    PackageManager,  // archive.ubuntu.com, pypi.org, crates.io
    SourceControl,   // github.com, gitlab.com
    ApiEndpoint,     // api.anthropic.com
    DnsInfra,        // dns.google
    AdNetwork,       // doubleclick.net
    Analytics,       // google-analytics.com
    Unknown,
}
```

Rules are suffix-based with exact-match priority:

```rust
pub struct CategoryRule {
    /// Domain pattern: exact ("crates.io") or suffix ("*.ubuntu.com")
    pub pattern: String,
    /// Category to assign
    pub category: DomainCategory,
}

impl CategoryPolicy {
    pub fn from_rules(rules: Vec<CategoryRule>) -> Self;
    pub fn default() -> Self; // built-in rule set
}
```

The default rule set is a starting point. It is not comprehensive — the
host runtime should extend or replace it based on the project's needs.

### Category → Action Mapping

Default behavior per category:

| Category | Action | Rationale |
|----------|--------|-----------|
| PackageManager | Allow | Expected dev workflow |
| SourceControl | Allow | Expected dev workflow |
| ApiEndpoint | Allow | Expected dev workflow |
| DnsInfra | Allow | Required for resolution |
| AdNetwork | Deny | Not expected in dev VM |
| Analytics | Deny | Not expected in dev VM |
| Unknown | Allow (confidence: 0.5) | Uncertain — deny_threshold or correlator decides |

The mapping is configurable — the host runtime can change Unknown → Deny
for stricter environments.

## Components and Testing

| Component | Test approach |
|-----------|--------------|
| `PacketParser` | Unit test: parse known-good DNS/TCP packets, verify extracted fields |
| DNS name decompression | Unit test: compressed labels, long names, edge cases |
| Response forging (NXDOMAIN) | Unit test: forge response, re-parse, verify NXDOMAIN |
| Response forging (TCP RST) | Unit test: forge RST, verify flags + sequence numbers |
| `DnsReverseMap` | Unit test: insert, lookup, TTL expiry |
| Chain evaluation | Unit test: Deny short-circuits, Allow min-confidence propagation, reason from weakest link |
| `LoggingPolicy` | Unit test: always Allow, tracing events emitted |
| `CategoryPolicy` | Unit test: suffix matching, known domains, Unknown handling |
| TX interceptor | Integration test: denied DNS → forged NXDOMAIN in rx channel |
| RX interceptor | Integration test: DNS response → reverse map updated |
| NoPolicy optimization | Build test: binary size delta ≈ 0 vs no-policy-support |
| End-to-end deny | Integration test: denied domain → guest sees NXDOMAIN |

## Alternatives Considered

### Alternative 1: Custom DNS Resolver (proxy approach)

Run a local DNS resolver on the host that `motlie-vnet` points to.

**Pros**: clean separation, standard DNS protocol
**Cons**: extra process, extra latency, doesn't cover TCP connections,
doesn't have guest-process context

### Alternative 2: eBPF/XDP in libslirp

Hook into libslirp's internal packet processing via eBPF.

**Pros**: zero-copy, kernel-level performance
**Cons**: requires kernel features, platform-dependent, violates the
rootless/no-capability design goal

### Alternative 3: Transparent proxy (e.g. mitmproxy)

Route all guest traffic through a proxy.

**Pros**: full application-layer visibility, TLS inspection
**Cons**: requires CA trust, breaks certificate pinning, complex setup,
violates the embeddable library goal

### Alternative 4: Box<dyn EgressPolicy> dynamic dispatch

Use trait objects for runtime polymorphism.

**Pros**: simpler type signatures, runtime-configurable chains
**Cons**: vtable dispatch per packet in the hot loop, heap allocation
for the policy chain, cannot be optimized away for NoPolicy

### Chosen: Generic type parameter with trait-level chaining

**Why**: zero-cost when unused (NoPolicy compiled out), monomorphized
chain calls in the hot loop, no heap allocation, composable via
`.chain()` on the trait. The host runtime knows all policy types at
compile time — dynamic dispatch adds cost without benefit.

---

## Appendix: How Chain Evaluation Works Internally

*This section is technical background for implementers. Developers using
the policy API never interact with these internals — they only call
`.chain()` on the trait and get back an `impl EgressPolicy`.*

When a developer writes:

```rust
let policy = LoggingPolicy.chain(CategoryPolicy::default()).chain(ExfilDetector::new(cfg));
```

The compiler constructs a nested type. Conceptually:

```
policy: Chain<LoggingPolicy, Chain<CategoryPolicy, ExfilDetector>>
```

The actual struct name is internal to the crate and not exported. Each
link holds `current` (this policy) and `next` (the rest of the chain).
The struct implements `EgressPolicy` by calling `self.current` first,
then `self.next`, with these rules:

- `Deny` from current → return immediately (short-circuit), carry its
  reason and confidence
- `Allow` from current → call next:
  - If next says `Deny` → return that Deny
  - If next says `Allow` → return `Allow` with
    `min(current.confidence, next.confidence)` and the reason from
    whichever had the lower confidence (weakest link)

For `on_tcp_flow` (and vfs `on_complete`), there is **no short-circuit**
— all policies in the chain see every flow report, because the flow
already exists and observation should not be suppressed.

Because the type is fully generic (no `Box<dyn>`), the compiler
monomorphizes the entire chain at compile time. Each callback inlines
through the nesting — zero vtable dispatch, zero heap allocation, zero
dynamic method resolution on the hot path.

When `P = NoPolicy`, the compiler sees that all methods return
`PolicyAction::Allow { confidence: 1.0 }` and eliminates the entire
chain evaluation as dead code.
