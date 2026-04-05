# motlie-vnet Policy Engine: Delivery Plan

Derived from [DESIGN-policy.md](./DESIGN-policy.md). Implements frame-level
DNS/TCP interception with extensible policy callbacks.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-05 | @claude-tl | Add PolicyReason, on_tcp_flow, FlowTracker, SNI parsing tasks. Renumber Phase 3 sections |
| 2026-04-05 | @claude-tl | Update: generic policy type flow, ChainedPolicy tasks, NoPolicy zero-cost tasks, logging-as-policy |
| 2026-04-05 | @claude-tl | Initial PLAN: 4 phases from packet parsing to composed integration |

## Status

All phases pending. No implementation yet.

---

## Phase 1: Packet Parser and DNS Codec

Design references: [Packet Parsing](./DESIGN-policy.md), [Response Forging](./DESIGN-policy.md)

### 1.1 Packet Parser (`src/policy/packet.rs`)

- [ ] 1.1.1 Create `src/policy/` module with `mod.rs`, `packet.rs`.
  Add `pub mod policy;` to `lib.rs`.

- [ ] 1.1.2 Implement `parse_frame(frame: &[u8]) -> ParsedPacket`.
  Parse Ethernet header (14 bytes) → IPv4 header (variable, IHL field) →
  protocol dispatch:
  - UDP + port 53 → DNS query or response
  - TCP + SYN flag → `TcpSyn`
  - Everything else → `Other`
  Design ref: [Packet Parsing](./DESIGN-policy.md)

- [ ] 1.1.3 Implement DNS name extraction from query section.
  Handle label encoding (length-prefixed segments), name compression
  (pointer labels with 0xC0 prefix). Return the FQDN as a String.
  ```rust
  fn parse_dns_name(data: &[u8], offset: usize) -> Option<(String, usize)>
  ```

- [ ] 1.1.4 Implement DNS response answer parsing.
  Extract A records (type 1): name, TTL, IPv4 address.
  Return `Vec<DnsAnswer>`.

- [ ] 1.1.5 Add test: parse a known DHCP discover frame → `Other`.
- [ ] 1.1.6 Add test: parse a DNS query for "example.com" → `DnsQuery { domain: "example.com", ... }`.
- [ ] 1.1.7 Add test: parse a DNS response with A record → `DnsResponse` with resolved IPs.
- [ ] 1.1.8 Add test: parse a TCP SYN to 93.184.216.34:443 → `TcpSyn { dst_ip, dst_port: 443, ... }`.
- [ ] 1.1.9 Add test: DNS name with compression pointers.
- [ ] 1.1.10 Add test: malformed/truncated packets → `Other` (no panic).

### 1.2 Response Forging (`src/policy/forge.rs`)

- [ ] 1.2.1 Implement `forge_dns_nxdomain(original_query: &[u8]) -> Vec<u8>`.
  Construct a complete Ethernet + IP + UDP + DNS response frame:
  - Copy transaction ID from query
  - Set QR=1 (response), RCODE=3 (NXDOMAIN)
  - Swap src/dst MAC, src/dst IP, src/dst port
  - Compute IP checksum
  Design ref: [Response Forging](./DESIGN-policy.md)

- [ ] 1.2.2 Implement `forge_tcp_rst(original_syn: &[u8]) -> Vec<u8>`.
  Construct Ethernet + IP + TCP RST:
  - Set ACK flag, seq=0, ack=syn_seq+1
  - Swap src/dst MAC, IP, port
  - Compute IP + TCP checksums
  Design ref: [Response Forging](./DESIGN-policy.md)

- [ ] 1.2.3 Implement `parse_tls_sni(payload: &[u8]) -> Option<String>`.
  Parse TLS ClientHello to extract the SNI extension hostname.
  Only parses the first data packet on port 443 connections.
  Design ref: [SNI Extraction](./DESIGN-policy.md)

- [ ] 1.2.4 Add test: forge NXDOMAIN from a real DNS query, re-parse → verify NXDOMAIN.
- [ ] 1.2.5 Add test: forge RST from a real TCP SYN, re-parse → verify RST+ACK flags.
- [ ] 1.2.6 Add test: forged responses have correct checksums (validate with `ip_checksum`).
- [ ] 1.2.7 Add test: extract SNI from a real TLS ClientHello.
- [ ] 1.2.8 Add test: non-TLS data on port 443 → SNI returns None (no panic).

---

## Phase 2: Policy Trait and DNS Reverse Map

Design references: [Policy Callback API](./DESIGN-policy.md), [Domain→IP Reverse Map](./DESIGN-policy.md)

### 2.1 Policy Types and Trait (`src/policy/mod.rs`)

- [ ] 2.1.1 Define `DnsQueryContext`, `TcpConnectContext`, `DnsResponseContext` structs.
  Design ref: [Policy Context Types](./DESIGN-policy.md)

- [ ] 2.1.2 Define `PolicyReason` enum: `Category(DomainCategory)`, `Custom(Cow<'static, str>)`.
  Implement `From<&'static str>`, `From<String>`, `From<DomainCategory>`.
  Design ref: [PolicyReason](./DESIGN-policy.md)

- [ ] 2.1.3 Define `PolicyAction` enum using `PolicyReason`:
  `Allow { reason: Option<PolicyReason> }`, `Deny { reason: PolicyReason }`,
  `Observe { reason: PolicyReason }`.

- [ ] 2.1.4 Define `EgressPolicy` trait with default allow-all impls and `chain()` method.
  Methods: `on_dns_query()`, `on_dns_response()`, `on_tcp_connect()`, `on_tcp_flow()`, `chain()`.
  All callbacks take `&self` — stateful policies use interior mutability.
  Trait bound: `Send + 'static`.
  `chain()` returns `ChainedPolicy<Self, N>` — generic, no Box/dyn.
  Design ref: [Core Trait](./DESIGN-policy.md), [Chaining](./DESIGN-policy.md)

- [ ] 2.1.4 Implement `ChainedPolicy<A, B>` struct with `EgressPolicy` impl.
  Evaluation: first Deny wins (short-circuit), Observe is sticky, Allow defers.
  Design ref: [Chaining](./DESIGN-policy.md)

- [ ] 2.1.5 Implement `NoPolicy` — zero-cost default, all methods return `Allow { reason: None }`.
  When used as the generic parameter, the compiler eliminates the interceptor entirely.
  Design ref: [Zero-Cost NoPolicy Default](./DESIGN-policy.md)

- [ ] 2.1.6 Implement `LoggingPolicy` — always returns `Allow`, logs every decision
  via `tracing::info!` with structured fields. Participates in the chain like
  any other policy — it is not special-cased.
  Design ref: [Logging as a Policy](./DESIGN-policy.md)

- [ ] 2.1.7 Add test: `NoPolicy` returns `Allow` for all contexts.
- [ ] 2.1.8 Add test: `LoggingPolicy` returns `Allow` for all contexts.
- [ ] 2.1.9 Add test: `ChainedPolicy` — first Deny wins.
- [ ] 2.1.10 Add test: `ChainedPolicy` — Observe is sticky (not overridden by later Allow).
- [ ] 2.1.11 Add test: `ChainedPolicy` — Allow defers to next policy.
- [ ] 2.1.12 Add test: three-policy chain via `.chain().chain()` compiles and works.

### 2.2 DNS Reverse Map (`src/policy/reverse_map.rs`)

- [ ] 2.2.1 Implement `DnsReverseMap` with `insert(ip, domain, ttl)` and
  `lookup(ip) -> Option<&str>`.
  - TTL-based expiry: entries expire after their DNS TTL
  - Multiple IPs per domain (A records return multiple)
  - Multiple domains per IP (CDNs share IPs)
  Design ref: [Domain→IP Reverse Map](./DESIGN-policy.md)

- [ ] 2.2.2 Add test: insert + lookup returns domain.
- [ ] 2.2.3 Add test: expired TTL → lookup returns None.
- [ ] 2.2.4 Add test: multiple IPs for one domain.

### 2.3 Category Policy Reference Implementation (`src/policy/category.rs`)

- [ ] 2.3.1 Implement `CategoryPolicy` with a built-in domain classification:
  - PackageManager: `*.ubuntu.com`, `pypi.org`, `crates.io`, `npmjs.com`, `registry.yarnpkg.com`
  - SourceControl: `github.com`, `gitlab.com`, `bitbucket.org`
  - ApiEndpoint: `api.anthropic.com`, `api.openai.com`
  - DnsInfra: `dns.google`, `1.1.1.1.in-addr.arpa`
  - AdNetwork: `doubleclick.net`, `googlesyndication.com`
  - Analytics: `google-analytics.com`, `mixpanel.com`, `segment.io`
  - Unknown: everything else
  Design ref: [Intent-Based Policy Example](./DESIGN-policy.md)

- [ ] 2.3.2 Implement suffix/pattern matching for domain classification.
  Support exact match (`crates.io`) and suffix match (`*.ubuntu.com`).

- [ ] 2.3.3 `on_dns_query()`: Allow PackageManager/SourceControl/ApiEndpoint/DnsInfra,
  Deny AdNetwork/Analytics, Observe Unknown.

- [ ] 2.3.4 `on_tcp_connect()`: look up domain from reverse map, apply same
  category logic. If domain is None (direct IP access), return Observe.

- [ ] 2.3.5 Add test: known domains classified correctly.
- [ ] 2.3.6 Add test: suffix matching works (e.g. `security.ubuntu.com` → PackageManager).
- [ ] 2.3.7 Add test: unknown domain → Observe.
- [ ] 2.3.8 Add test: TCP connect with reverse-mapped domain → correct category.
- [ ] 2.3.9 Add test: TCP connect with raw IP (no DNS) → Observe.

---

## Phase 3: Interceptor Integration

Design references: [Interception Points](./DESIGN-policy.md), [Integration with VnetBackend](./DESIGN-policy.md)

### 3.1 TX Interceptor (`src/policy/interceptor.rs`)

- [ ] 3.1.1 Implement `TxInterceptor<P: EgressPolicy>` — generic over the policy type.
  Holds the policy, reverse map, and forging functions. Interface:
  ```rust
  impl<P: EgressPolicy> TxInterceptor<P> {
      /// Inspect a guest→host frame. Returns:
      /// - Some(response_frame) if denied (forged NXDOMAIN or RST to inject)
      /// - None if allowed (frame should proceed to slirp.input())
      fn intercept(&mut self, frame: &[u8]) -> Option<Vec<u8>>;
  }
  ```
  When `P = NoPolicy`, the compiler eliminates the parse + callback + forge
  code entirely.
  Design ref: [Generic Type Flow](./DESIGN-policy.md)

- [ ] 3.1.2 Wire `TxInterceptor` into `slirp_thread_main()`.
  Before `slirp.input(&frame)`, call `interceptor.intercept(&frame)`.
  If it returns `Some(response)`, skip `slirp.input()` and queue the
  response directly into `rx_sender`.
  Design ref: [TX path interception](./DESIGN-policy.md)

- [ ] 3.1.3 Add test: TX interceptor with deny-all DNS policy → DNS query produces
  forged NXDOMAIN in rx channel, frame not passed to slirp.

- [ ] 3.1.4 Add test: TX interceptor with deny-all TCP policy → TCP SYN produces
  forged RST in rx channel, frame not passed to slirp.

- [ ] 3.1.5 Add test: TX interceptor with allow-all → frame passed through, no
  forged response.

### 3.2 Flow Tracker (`src/policy/flow.rs`)

- [ ] 3.2.1 Implement `FlowTracker` — maintains per-connection state:
  ```rust
  struct FlowState {
      dst_ip: Ipv4Addr,
      dst_port: u16,
      domain: Option<String>,
      sni: Option<String>,       // extracted from first TX data packet
      source_port: u16,
      bytes_tx: u64,
      bytes_rx: u64,
      started: Instant,
      sni_checked: bool,         // only parse first TX packet for SNI
  }
  ```
  Keyed by `(src_port, dst_ip, dst_port)`. Created on SYN, updated on
  data packets, removed on FIN/RST.

- [ ] 3.2.2 On first TX data packet to port 443: call `parse_tls_sni()`,
  store in `FlowState.sni`.

- [ ] 3.2.3 Call `policy.on_tcp_flow()` periodically (every N packets or
  every T seconds) and once on flow close. If Deny, forge RST.

- [ ] 3.2.4 Add test: flow tracker creates state on SYN, removes on FIN.
- [ ] 3.2.5 Add test: SNI extracted from first TLS packet on port 443.
- [ ] 3.2.6 Add test: on_tcp_flow called with correct byte counts.

### 3.3 RX Interceptor

- [ ] 3.3.1 Implement RX-side DNS response observation in `SlirpHandler::send_packet()`.
  Parse DNS responses, update the reverse map, call `policy.on_dns_response()`.
  If Deny, rewrite frame to NXDOMAIN before queueing.

- [ ] 3.3.2 Add test: DNS response updates reverse map.
- [ ] 3.3.3 Add test: RX deny rewrites response to NXDOMAIN.

### 3.4 VnetConfig Integration

- [ ] 3.4.1 Make `VnetConfig<P: EgressPolicy = NoPolicy>` generic over the policy type.
  Builder method `.egress_policy(policy)` changes the generic parameter.
  Default: `NoPolicy` (zero-cost, interceptor compiled out).
  Design ref: [Generic Type Flow](./DESIGN-policy.md)

- [ ] 3.4.2 Make `VnetBackend<P>`, `slirp_thread_main<P>()` generic over P.
  Thread the policy through `start()` → slirp thread → `TxInterceptor<P>`
  + `FlowTracker` + RX interceptor.

- [ ] 3.4.3 Add `PolicyStats` struct to `VnetHandle`:
  ```rust
  pub struct PolicyStats {
      pub dns_allowed: u64,
      pub dns_denied: u64,
      pub dns_observed: u64,
      pub tcp_allowed: u64,
      pub tcp_denied: u64,
      pub tcp_observed: u64,
      pub unique_domains: usize,
  }
  ```

- [ ] 3.4.4 Add test: `VnetConfig<NoPolicy>` compiles, starts, and runs with
  zero interceptor overhead (existing tests continue to pass unchanged).
- [ ] 3.4.5 Add test: `VnetConfig` with `LoggingPolicy` → policy callbacks invoked.
- [ ] 3.4.6 Add test: `VnetConfig` with chained policy → chain evaluated correctly.

---

## Phase 4: End-to-End and Documentation

- [ ] 4.1.1 Add integration test: full slirp instance with CategoryPolicy.
  Feed DNS query for `archive.ubuntu.com` → allowed.
  Feed DNS query for `doubleclick.net` → denied (NXDOMAIN).
  Feed TCP SYN to known-good IP → allowed.

- [ ] 4.1.2 Add `libs/vnet/examples/policy_demo.rs`:
  - Start vnet with CategoryPolicy
  - Print policy decisions as they happen
  - Demonstrate deny/allow/observe

- [ ] 4.1.3 Update `libs/vnet/docs/DESIGN.md` with a cross-reference to the
  policy engine and how it integrates with the egress path.

- [ ] 4.1.4 Update PLAN.md (main) to reference the policy engine phases.

- [ ] 4.1.5 Add `libs/vnet/README.md` section on policy configuration.

---

## Delivery Order

1. **Phase 1** (1.1 → 1.2) — packet parsing + response forging (pure functions, no integration)
2. **Phase 2** (2.1 → 2.2 → 2.3) — policy trait + reverse map + reference implementation
3. **Phase 3** (3.1 → 3.2 → 3.3) — wire into the vnet backend
4. **Phase 4** — e2e tests, examples, docs

Phases 1 and 2 are independent of the vnet backend — pure library code
with comprehensive unit tests. Phase 3 wires into the existing two-thread
architecture. Phase 4 validates end-to-end.

## Dependencies

| Dependency | Type | Required for |
|------------|------|-------------|
| Existing `motlie-vnet` (Phases 1-2) | Crate | Phase 3+ |
| `EgressPolicy` trait impl | Host runtime | Phase 3+ (default provided) |
| No new system dependencies | — | — |
