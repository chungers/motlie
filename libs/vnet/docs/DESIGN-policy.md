# motlie-vnet Policy Engine: Observability and Egress Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-05 | @claude-tl | Redesign: fully generic policy (no Box/dyn), chainable via trait method, logging-as-policy, zero-cost NoPolicy default, honest intent derivation layering |
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
   logged reason), or observe-only (allow but flag).

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

### Core Trait

```rust
/// Policy decision returned by callbacks.
pub enum PolicyAction {
    /// Allow the request. Optionally attach a reason for audit logging.
    Allow { reason: Option<String> },
    /// Deny the request. The engine forges a failure response (NXDOMAIN
    /// for DNS, RST for TCP). The reason is logged.
    Deny { reason: String },
    /// Allow but flag for observation. The request proceeds normally but
    /// the engine logs it at a higher severity.
    Observe { reason: String },
}

/// Egress policy callbacks. All methods have default allow-all impls.
///
/// Callbacks run on the slirp thread. They must not block or perform I/O.
/// For async policy evaluation, return Observe and make the decision
/// asynchronously with a later connection teardown if needed.
pub trait EgressPolicy: Send + 'static {
    /// Called when the guest issues a DNS query.
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Called when a DNS response arrives from the host resolver.
    fn on_dns_response(&self, ctx: &DnsResponseContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Called when the guest initiates a TCP connection (SYN detected).
    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Chain another policy after this one.
    ///
    /// Evaluation order: `self` first, then `next`.
    /// - First `Deny` wins (short-circuit).
    /// - `Observe` is sticky — propagates even if `next` returns `Allow`.
    /// - `Allow` defers to `next`.
    fn chain<N: EgressPolicy>(self, next: N) -> ChainedPolicy<Self, N>
    where
        Self: Sized,
    {
        ChainedPolicy { current: self, next }
    }
}
```

### Chaining

Policies compose via the `chain()` method on the trait. The result is a
concrete generic type — no heap allocation, no vtable dispatch.

```rust
/// Two policies evaluated in sequence. First Deny wins, Observe is sticky.
pub struct ChainedPolicy<A, B> {
    current: A,
    next: B,
}

impl<A: EgressPolicy, B: EgressPolicy> EgressPolicy for ChainedPolicy<A, B> {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        match self.current.on_dns_query(ctx) {
            PolicyAction::Deny { reason } => PolicyAction::Deny { reason },
            PolicyAction::Observe { reason } => {
                match self.next.on_dns_query(ctx) {
                    PolicyAction::Deny { reason } => PolicyAction::Deny { reason },
                    _ => PolicyAction::Observe { reason },
                }
            }
            PolicyAction::Allow { .. } => self.next.on_dns_query(ctx),
        }
    }

    fn on_dns_response(&self, ctx: &DnsResponseContext) -> PolicyAction {
        // same pattern
        # ..
    }

    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        // same pattern
        # ..
    }
}
```

The compiler monomorphizes the chain:

```rust
let policy = LoggingPolicy::new()
    .chain(CategoryPolicy::default())
    .chain(my_runtime_policy);

// Concrete type at compile time:
// ChainedPolicy<LoggingPolicy, ChainedPolicy<CategoryPolicy, MyPolicy>>
// All calls inlined — zero dynamic dispatch in the hot loop.
```

### Zero-Cost NoPolicy Default

```rust
/// Zero-cost policy that allows everything. When used as the policy type,
/// the compiler optimizes the entire interceptor away — the hot path
/// compiles to the same code as if policy support did not exist.
pub struct NoPolicy;

impl EgressPolicy for NoPolicy {
    fn on_dns_query(&self, _: &DnsQueryContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }
    fn on_dns_response(&self, _: &DnsResponseContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }
    fn on_tcp_connect(&self, _: &TcpConnectContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }
}
```

### Logging as a Policy

Logging is not special — it's a regular `EgressPolicy` implementation
that always returns `Allow` but emits structured tracing events. It
participates in the chain like any other policy.

```rust
/// Logs every DNS query and TCP connection via tracing.
/// Always returns Allow — observe-only, no enforcement.
pub struct LoggingPolicy;

impl EgressPolicy for LoggingPolicy {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        tracing::info!(
            subsystem = "vnet-policy",
            domain = %ctx.domain,
            query_type = ?ctx.query_type,
            source_port = ctx.source_port,
            "DNS query"
        );
        PolicyAction::Allow { reason: None }
    }

    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        tracing::info!(
            subsystem = "vnet-policy",
            dst_ip = %ctx.dst_ip,
            dst_port = ctx.dst_port,
            domain = ?ctx.domain,
            source_port = ctx.source_port,
            "TCP connect"
        );
        PolicyAction::Allow { reason: None }
    }

    // on_dns_response: log resolved IPs
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
    pub timestamp: std::time::Instant,
}

pub struct TcpConnectContext {
    /// Destination IP address.
    pub dst_ip: std::net::Ipv4Addr,
    /// Destination port.
    pub dst_port: u16,
    /// Domain name that resolved to this IP, if known from a prior DNS
    /// lookup. None if the guest connected directly by IP.
    pub domain: Option<String>,
    /// Guest-side source port.
    pub source_port: u16,
    /// Timestamp.
    pub timestamp: std::time::Instant,
}

pub struct DnsResponseContext {
    /// The queried domain name.
    pub domain: String,
    /// Resolved IPv4 addresses.
    pub resolved_ips: Vec<std::net::Ipv4Addr>,
    /// TTL from the DNS response.
    pub ttl: u32,
    /// Timestamp.
    pub timestamp: std::time::Instant,
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
report which PID opened which socket. This context flows into
`TcpConnectContext` as an optional field:

```rust
pub struct TcpConnectContext {
    // ... existing fields ...
    /// Guest process name that initiated this connection, if reported
    /// by the guest agent. Enables true intent-based decisions:
    /// "apt" connecting to ubuntu.com = Allow
    /// "unknown-binary" connecting to ubuntu.com = Observe
    pub guest_process: Option<String>,
}
```

This is the long-term target. The policy trait is designed to accommodate
it — `guest_process` is `Option`, so v1 policies work without it, and
v2 policies can use it when available.

### v3: Project Manifest Correlation (future)

The host runtime could parse `Cargo.toml`, `package.json`, `requirements.txt`
to derive an expected dependency set, then auto-generate an allowlist of
domains those dependencies need (crates.io, npmjs.com, pypi.org, etc.).
Connections outside the manifest are flagged.

This is a host-runtime concern, not a vnet concern — it would be
implemented as a custom `EgressPolicy` that reads the manifest.

## Generic Type Flow

The policy type parameter flows through the entire stack:

```
VnetConfig<P>           // P = NoPolicy | LoggingPolicy | ChainedPolicy<...>
  → VnetBackend<P>
    → slirp_thread_main<P>()
      → TxInterceptor<P>   // owns the policy, calls on_dns_query/on_tcp_connect
      → RxInterceptor<P>   // borrows the policy, calls on_dns_response
```

`VnetConfig` defaults to `NoPolicy`:

```rust
impl VnetConfig<NoPolicy> {
    pub fn builder() -> VnetConfigBuilder<NoPolicy> { ... }
}

impl<P: EgressPolicy> VnetConfigBuilder<P> {
    pub fn egress_policy<Q: EgressPolicy>(self, policy: Q) -> VnetConfigBuilder<Q> {
        // Replaces the policy type — builder returns a new generic variant
        VnetConfigBuilder {
            policy,
            socket_path: self.socket_path,
            // ... copy other fields
        }
    }
}
```

When `P = NoPolicy`, the compiler sees that all interceptor methods
return `Allow { reason: None }` and optimizes away the parsing, the
reverse map lookup, and the response forging code. The slirp thread
loop compiles to the same machine code as the no-policy path.

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
- `on_dns_query()` is `{ PolicyAction::Allow { reason: None } }`
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

### Observe-only (logging, no enforcement)

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
| Unknown | Observe | Flag for review, don't block |

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
| `ChainedPolicy` | Unit test: Deny short-circuits, Observe sticky, Allow defers |
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
