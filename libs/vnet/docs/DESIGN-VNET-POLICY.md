# motlie-vnet Policy Engine: Observability and Egress Control

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-05 | @claude-tl | Add DNS exfiltration detection use case, label_lengths/name_wire_len in DnsQueryContext, defense-in-depth rationale |
| 2026-04-05 | @claude-tl | Add errno to PolicyAction, predefined errno pairs, OTel logging, EventSinkPolicy, cross-stack alignment with vfs, chain returns impl EgressPolicy |
| 2026-04-05 | @claude-tl | Add PolicyReason (Cow-backed enum), on_tcp_flow with flow metadata + SNI, exfiltration/C2/fronting use cases, honest DPI limitations, &self + interior mutability guidance |
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
/// Policy decision. Shared shape with motlie-vfs's FsPolicy.
pub enum PolicyAction {
    /// Allow the request. Optionally attach a reason for audit logging.
    Allow { reason: Option<PolicyReason> },
    /// Deny the request. `errno` gives the policy a semantic hook into
    /// *how* the denial manifests (NXDOMAIN, RST, etc.). The interceptor
    /// maps errno to the appropriate forged response.
    Deny { errno: i32, reason: PolicyReason },
    /// Allow but flag for observation. The request proceeds normally but
    /// the engine logs it at a higher severity.
    Observe { reason: PolicyReason },
}
```

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
/// For async policy evaluation, return `Observe` and make the decision
/// asynchronously with a later connection teardown if needed.
pub trait EgressPolicy: Send + 'static {
    /// Called when the guest issues a DNS query. Return Deny to prevent
    /// the query from reaching the host resolver (forged NXDOMAIN).
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Called when a DNS response arrives from the host resolver,
    /// before it is forwarded to the guest.
    fn on_dns_response(&self, ctx: &DnsResponseContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
    }

    /// Called when the guest initiates a TCP connection (SYN detected).
    /// Return Deny to forge a RST — no host-side socket is created.
    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        PolicyAction::Allow { reason: None }
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
        PolicyAction::Allow { reason: None }
    }

    /// Chain another policy after this one.
    ///
    /// Evaluation order: `self` first, then `next`.
    /// - First `Deny` wins (short-circuit).
    /// - `Observe` is sticky — propagates even if `next` returns `Allow`.
    /// - `Allow` defers to `next`.
    fn chain<N: EgressPolicy>(self, next: N) -> impl EgressPolicy
    where
        Self: Sized,
    {
        ChainedEgressPolicy { current: self, next }
    }
}
```

### Chaining

Policies compose via the `chain()` method on the trait. The result is an
opaque `impl EgressPolicy` — the internal `ChainedEgressPolicy<A, B>`
struct is not part of the public API. No heap allocation, no vtable dispatch.

Evaluation rules (same as `motlie-vfs`'s `FsPolicy`):
- First `Deny` wins (short-circuit)
- `Observe` is sticky — propagates even if next policy returns Allow
- `Allow` defers to next policy

```rust
let policy = LoggingPolicy
    .chain(CategoryPolicy::default())
    .chain(my_runtime_policy);

// Opaque type: impl EgressPolicy
// Compiler monomorphizes all calls — zero dynamic dispatch in the hot loop.
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
        PolicyAction::Allow { reason: None }
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
        PolicyAction::Allow { reason: None }
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
        PolicyAction::Allow { reason: None }
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
- `unknown-binary` connecting to `ubuntu.com` → Observe (unexpected process)

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
    PolicyAction::Allow { reason: None }
}
```

**C2 beacon detection**: a flow with periodic small tx/rx bursts to an
Unknown domain over a long duration (>30min) matches a command-and-control
pattern. The policy can observe and alert.

**Bandwidth accounting**: track bytes per domain/category for reporting.
`LoggingPolicy` can aggregate flow stats and emit periodic summaries.

**SNI verification**: if the DNS reverse map says domain X but the TLS
SNI says domain Y, the connection may be using domain fronting. The
policy can flag the mismatch:

```rust
fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
    if let (Some(dns_domain), Some(sni)) = (&ctx.domain, &ctx.sni) {
        if dns_domain != sni {
            return PolicyAction::Observe {
                reason: PolicyReason::Custom(
                    format!("SNI mismatch: DNS={}, SNI={}", dns_domain, sni).into()
                ),
            };
        }
    }
    PolicyAction::Allow { reason: None }
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

        PolicyAction::Allow { reason: None }
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

**Implementation using the policy API:**

```rust
/// Shannon entropy of a byte string (bits per character).
/// Returns 0.0 for empty input.
fn shannon_entropy(s: &str) -> f64 {
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

/// Extract the base domain (last two labels) from an FQDN.
/// "a.b.c.attacker.com" → "attacker.com"
fn extract_base_domain(domain: &str) -> String {
    let labels: Vec<&str> = domain.split('.').collect();
    if labels.len() >= 2 {
        labels[labels.len() - 2..].join(".")
    } else {
        domain.to_string()
    }
}
```

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

## EventSinkPolicy (generic event producer)

A chainable policy that sends context to a caller-supplied closure.
Always returns Allow — observe-only, no enforcement. Enables cross-stack
event correlation without coupling vnet to vfs.

```rust
pub struct EventSinkPolicy<F> {
    on_event: F,
}

impl<F> EventSinkPolicy<F>
where
    F: Fn(NetEvent) + Send + 'static,
{
    pub fn new(on_event: F) -> Self {
        Self { on_event }
    }
}

/// Events emitted by the sink policy.
pub enum NetEvent {
    DnsQuery(DnsQueryContext),
    DnsResponse(DnsResponseContext),
    TcpConnect(TcpConnectContext),
    TcpFlow(TcpFlowContext),
}
```

Usage at the host runtime (which depends on both vfs and vnet):

```rust
// Host runtime defines its own unified event:
enum RuntimeEvent {
    Fs(motlie_vfs::FsOpContext),
    Net(motlie_vnet::NetEvent),
}

let (tx, rx) = mpsc::channel::<RuntimeEvent>();

// vnet side:
let tx_net = tx.clone();
let net_policy = LoggingPolicy
    .chain(CategoryPolicy::default())
    .chain(EventSinkPolicy::new(move |evt| {
        let _ = tx_net.send(RuntimeEvent::Net(evt));
    }));
```

Neither crate depends on the other. The host runtime owns the unified
enum and the channel. Each crate provides `EventSinkPolicy<F>` with a
generic `Fn` — it doesn't know about the other crate's types.

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
