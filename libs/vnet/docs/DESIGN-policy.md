# motlie-vnet Policy Engine: Observability and Egress Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
   behind an action (e.g. "this is a package manager doing apt update"
   vs "this is an unknown binary phoning home"), not just the domain name.
   The engine provides context; the policy callback decides.

4. **Extensibility**: the policy engine is a set of Rust trait callbacks.
   The host runtime supplies the policy implementation. `motlie-vnet`
   provides default implementations (allow-all with logging, and a
   category-based reference policy).

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
│  │ TX Policy Interceptor               │        │
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
│  │ RX Policy Interceptor               │        │
│  │                                     │        │
│  │ DNS response (UDP port 53)?        │        │
│  │   → parse response, extract domain │        │
│  │     + resolved IPs                  │        │
│  │   → call policy.on_dns_response()  │        │
│  │   → if Deny: drop or rewrite to   │        │
│  │     NXDOMAIN                        │        │
│  │   → update domain→IP reverse map  │        │
│  │                                     │        │
│  │ TCP data?                          │        │
│  │   → look up dst IP in reverse map │        │
│  │   → enrich with domain context    │        │
│  │   → call policy.on_tcp_data()     │        │
│  │     (observe only — no modification)│        │
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

## Policy Callback API

### Core Trait

```rust
/// Context provided to policy callbacks for decision-making.
pub struct DnsQueryContext {
    /// The queried domain name (e.g. "archive.ubuntu.com").
    pub domain: String,
    /// DNS record type (A, AAAA, CNAME, etc.).
    pub query_type: DnsQueryType,
    /// Guest-side source port (correlates with the requesting process
    /// if the host runtime has process visibility via vsock).
    pub source_port: u16,
    /// Timestamp of the query.
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

/// Policy callbacks implemented by the host runtime.
///
/// All methods have default implementations that allow everything with
/// logging (observe-all). Override specific methods to enforce policy.
///
/// Callbacks are invoked on the slirp thread. They must not block or
/// perform I/O — compute a decision and return immediately. For async
/// policy evaluation, return Observe and make the decision asynchronously
/// with a later connection teardown if needed.
pub trait EgressPolicy: Send + 'static {
    /// Called when the guest issues a DNS query. Return Deny to prevent
    /// the query from reaching the host resolver (forged NXDOMAIN).
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Called when a DNS response is received from the host resolver,
    /// before it is forwarded to the guest. Return Deny to rewrite the
    /// response to NXDOMAIN.
    fn on_dns_response(&self, ctx: &DnsResponseContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Called when the guest initiates a TCP connection (SYN detected).
    /// The domain field is populated from the DNS reverse map if the
    /// guest resolved this IP via DNS. Return Deny to forge a RST.
    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }
}
```

### Intent-Based Policy Example

The callback API provides context — the policy implementation decides
based on intent. Example: a category-based policy that classifies domains
by purpose:

```rust
pub enum DomainCategory {
    PackageManager,  // archive.ubuntu.com, pypi.org, crates.io, npmjs.com
    SourceControl,   // github.com, gitlab.com, bitbucket.org
    ApiEndpoint,     // api.anthropic.com, api.openai.com
    DnsInfra,        // dns.google, 1.1.1.1
    AdNetwork,       // doubleclick.net, googlesyndication.com
    Analytics,       // google-analytics.com, mixpanel.com
    Unknown,         // not categorized
}

impl EgressPolicy for CategoryPolicy {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        let category = self.categorize(&ctx.domain);
        match category {
            DomainCategory::PackageManager
            | DomainCategory::SourceControl
            | DomainCategory::ApiEndpoint
            | DomainCategory::DnsInfra => PolicyAction::Allow {
                reason: Some(format!("{:?}", category)),
            },
            DomainCategory::AdNetwork
            | DomainCategory::Analytics => PolicyAction::Deny {
                reason: format!("blocked {:?}: {}", category, ctx.domain),
            },
            DomainCategory::Unknown => PolicyAction::Observe {
                reason: format!("unknown domain: {}", ctx.domain),
            },
        }
    }
}
```

This is not the only policy shape — the trait is open. The host runtime
could implement:
- A learning-mode policy that observes all traffic for N minutes, then
  locks down to only the domains seen during learning
- A prompt-based policy that asks the user "allow git clone from X?"
  (via the REPL) on first access to an unknown domain
- An allowlist derived from a project manifest (Cargo.toml dependencies
  → crates.io, github.com)

### Domain→IP Reverse Map

DNS interception populates a reverse map (`HashMap<Ipv4Addr, DomainInfo>`)
so TCP connection decisions have domain context. Without this, TCP-level
policy would only see raw IP addresses.

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
    /// Record a DNS resolution.
    fn insert(&mut self, ip: Ipv4Addr, domain: String, ttl: u32) { ... }

    /// Look up the domain that resolved to this IP.
    /// Returns None if the IP was never resolved via DNS (direct IP access)
    /// or if the TTL has expired.
    fn lookup(&self, ip: &Ipv4Addr) -> Option<&str> { ... }
}
```

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
    Other, // pass through without policy evaluation
}

pub struct DnsAnswer {
    pub ip: Ipv4Addr,
    pub ttl: u32,
}
```

### Response Forging

When policy denies a request, the interceptor forges a guest-visible
response:

- **DNS deny**: construct an NXDOMAIN response with the original query's
  transaction ID, matching the expected DNS response format. Inject into
  the rx path (bypassing libslirp entirely).
- **TCP deny**: construct a TCP RST packet with correct sequence numbers
  derived from the SYN. Inject into the rx path.

Both require constructing valid Ethernet + IP + protocol headers with
correct checksums.

## Integration with VnetBackend

### Configuration

```rust
let policy = Arc::new(MyPolicy::new());

let config = VnetConfig::builder()
    .socket_path("/tmp/motlie-vnet-0.sock")
    .egress_policy(policy)  // optional — default is allow-all with logging
    .build()?;
```

The `EgressPolicy` trait object is `Arc<dyn EgressPolicy>` — shared
between the slirp thread (where interception runs) and the host runtime
(which may update policy state).

### Threading

Policy callbacks run on the slirp thread. They must be non-blocking.
The `EgressPolicy` trait requires `Send` (moved to slirp thread) but
not `Sync` — the slirp thread is the sole caller.

If the host runtime needs to update policy state (e.g. add a domain to
an allowlist), it communicates via `Arc<AtomicBool>` flags or
`Arc<RwLock<PolicyState>>` — the policy implementation manages its own
synchronization.

### Performance Impact

Per-packet overhead:
- **Ethernet/IP/TCP header parsing**: ~50ns (fixed offsets, no allocation)
- **DNS name parsing**: ~200ns (label decompression, string allocation)
- **Policy callback**: depends on implementation (HashMap lookup ~50ns)
- **Response forging (deny only)**: ~500ns (construct + checksum)

Total: ~300ns per allowed packet, ~800ns per denied packet. Negligible
compared to libslirp's per-packet overhead (~1-10us) and syscall costs.

## Observability

### Structured Logging

All policy decisions are logged via `tracing` with structured fields:

```rust
tracing::info!(
    subsystem = "vnet-policy",
    domain = %ctx.domain,
    action = "deny",
    reason = %reason,
    query_type = ?ctx.query_type,
    source_port = ctx.source_port,
    "DNS query denied"
);
```

### Metrics (Future)

The interceptor maintains counters:
- `dns_queries_total` (by action: allow/deny/observe)
- `tcp_connections_total` (by action)
- `dns_domains_seen` (unique count)

Exposed via a `PolicyStats` struct accessible from `VnetHandle`.

## Components and Testing

| Component | Test approach |
|-----------|--------------|
| `PacketParser` | Unit test: parse known-good DNS/TCP packets, verify extracted fields |
| DNS name decompression | Unit test: compressed labels, long names, edge cases |
| Response forging (NXDOMAIN) | Unit test: forge response, re-parse, verify NXDOMAIN |
| Response forging (TCP RST) | Unit test: forge RST, verify flags + sequence numbers |
| `DnsReverseMap` | Unit test: insert, lookup, TTL expiry |
| TX interceptor | Integration test: inject DNS query frame, verify policy callback called |
| RX interceptor | Integration test: inject DNS response, verify reverse map updated |
| End-to-end deny | Integration test: denied domain → guest sees NXDOMAIN |
| CategoryPolicy | Unit test: known domains categorized correctly |

## Alternatives Considered

### Alternative 1: Custom DNS Resolver (proxy approach)

Run a local DNS resolver on the host that `motlie-vnet` points to
(via `dns_ipv4`). The resolver applies policy before forwarding.

**Pros**: clean separation, standard DNS protocol, works with any policy engine
**Cons**: extra process, extra latency (local socket hop), doesn't cover
TCP connections, doesn't have guest-process context

### Alternative 2: eBPF/XDP in libslirp

Hook into libslirp's internal packet processing via eBPF.

**Pros**: zero-copy, kernel-level performance
**Cons**: requires kernel features, platform-dependent, libslirp doesn't
expose eBPF hooks, violates the rootless/no-capability design goal

### Alternative 3: Transparent proxy (e.g. mitmproxy)

Route all guest traffic through a proxy.

**Pros**: full application-layer visibility, TLS inspection possible
**Cons**: requires CA trust, breaks certificate pinning, adds significant
latency, complex setup, violates the embeddable library goal

### Chosen: Frame-level interception in the Rust layer

**Why**: works within the existing two-thread architecture, no extra
processes, no kernel features, no TLS breakage. Policy callbacks keep
the engine extensible without baking in specific rules. The domain→IP
reverse map bridges the DNS→TCP context gap.
