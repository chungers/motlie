# motlie-policy: Shared Detection Primitives

Reusable analysis functions and detection heuristics for `motlie-vfs`
and `motlie-vnet` policy implementations. This crate contains no
policy logic itself — it provides the building blocks that policy
implementations compose.

## Modules

### `entropy` — Shannon Entropy Analysis

Measures randomness of strings to distinguish natural-language hostnames
and file paths from encoded/encrypted data (base64, hex, compressed blobs).

**Use cases:**
- DNS exfiltration: high-entropy subdomain labels carrying encoded payloads
- Filesystem staging: high-entropy filenames indicating obfuscated payloads

**Feasibility with vnet policy API:** Fully supported. `DnsQueryContext`
provides `domain`, `label_lengths`, `name_wire_len`. A policy calls
`shannon_entropy()` on each label and compares against a threshold.
No additional API surface needed.

### `domain` — Domain Name Utilities

Base domain extraction, suffix matching, label parsing.

**Use cases:**
- Rate tracking per base domain (DNS exfil detection)
- Category matching (suffix-based domain classification)

### `ratio` — Byte Ratio Analysis (placeholder)

Detects asymmetric transfer patterns by computing the outbound/inbound
byte ratio over a flow's lifetime.

**Use cases:**
- Data exfiltration: a flow with bytes_tx >> bytes_rx to an Unknown
  domain (ratio > 10:1) suggests bulk upload, not normal request/response.
- Download anomaly: bytes_rx >> bytes_tx is normal for package downloads
  but suspicious if the destination is uncategorized.

**Feasibility with vnet policy API:** Fully supported. `TcpFlowContext`
provides `bytes_tx` and `bytes_rx`. A policy computes the ratio in
`on_tcp_flow()`:

```rust
let ratio = ctx.bytes_tx as f64 / (ctx.bytes_rx.max(1) as f64);
if ratio > 10.0 && ctx.bytes_tx > 1_000_000 {
    // High outbound ratio + significant volume → exfil signal
}
```

No additional API surface needed. The periodic `on_tcp_flow()` callback
gives the policy running totals at each report interval.

### `duration` — Flow Duration Analysis (placeholder)

Detects persistent low-and-slow data leaks by tracking flow lifetimes
and flagging connections that remain open for unusually long periods
with small, periodic data transfers.

**Use cases:**
- Slow exfiltration: a connection that sends 1KB every 30 seconds for
  hours. Individual transfers are too small to trigger byte-ratio alerts,
  but the cumulative volume and duration are suspicious.
- Persistent tunnels: SSH or VPN tunnels to unexpected destinations that
  stay open indefinitely.

**Feasibility with vnet policy API:** Fully supported. `TcpFlowContext`
provides `duration` (time since SYN) and `bytes_tx`/`bytes_rx`. A policy
can compute bytes-per-second and flag low-rate long-lived flows:

```rust
if ctx.duration > Duration::from_secs(3600) {
    let tx_rate = ctx.bytes_tx as f64 / ctx.duration.as_secs_f64();
    if tx_rate > 0.0 && tx_rate < 1000.0 {
        // Slow trickle over > 1 hour → low-and-slow leak
    }
}
```

No additional API surface needed. The `is_close` field in `TcpFlowContext`
allows the policy to trigger final analysis when the flow ends.

### `beacon` — Heartbeat / Beacon Detection (placeholder)

Detects periodic communication patterns characteristic of command-and-control
(C2) implants. Beacons typically connect to a controller at regular intervals
(e.g. every 60s ± jitter) with small, fixed-size payloads.

**Use cases:**
- C2 detection: periodic small bursts (100-500 bytes) at regular intervals
  to an Unknown domain.
- Keep-alive detection: distinguishing legitimate keep-alives (to known
  services) from suspicious beaconing (to unknown infrastructure).

**Feasibility with vnet policy API:** Partially supported. `TcpFlowContext`
provides cumulative `bytes_tx`/`bytes_rx` and `duration`, but NOT
per-interval timing (individual packet timestamps or inter-arrival times).

To detect periodicity, a stateful policy would need to track flow reports
over time and compute inter-report deltas:

```rust
struct BeaconDetector {
    // flow_key → Vec<(timestamp, bytes_tx_delta)>
    flow_history: RwLock<HashMap<FlowKey, Vec<(Instant, u64)>>>,
}

impl EgressPolicy for BeaconDetector {
    fn on_tcp_flow(&self, ctx: &TcpFlowContext) -> PolicyAction {
        let key = (ctx.source_port, ctx.dst_ip, ctx.dst_port);
        let mut history = self.flow_history.write().unwrap();
        let entries = history.entry(key).or_default();
        entries.push((ctx.timestamp, ctx.bytes_tx));

        if entries.len() >= 10 {
            let intervals: Vec<f64> = entries.windows(2)
                .map(|w| (w[1].0 - w[0].0).as_secs_f64())
                .collect();
            let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let variance = intervals.iter()
                .map(|i| (i - mean).powi(2))
                .sum::<f64>() / intervals.len() as f64;
            let cv = variance.sqrt() / mean;  // coefficient of variation

            if cv < 0.3 && mean > 10.0 && mean < 300.0 {
                // Low variance (regular interval), 10-300s period → beacon
            }
        }
        PolicyAction::Allow { reason: None }
    }
}
```

**API gap:** The `on_tcp_flow()` report interval is controlled by the
interceptor, not the policy. If reports are too infrequent, the policy
cannot measure inter-arrival times accurately. Future consideration:
make the report interval configurable, or add a per-packet timestamp
callback for flow metadata policies.

### `first_seen` — First Destination Tracking (placeholder)

Tracks the first time each IP address or domain is observed, enabling
policies that treat "never seen before" destinations with higher scrutiny
than established ones.

**Use cases:**
- Novel destination alerting: the first connection to a never-before-seen
  IP or domain triggers Observe, while subsequent connections to the same
  destination are Allow. Useful for detecting connections to newly-spun-up
  infrastructure (common in attacks, rare in normal dev workflows).
- Baseline learning: during an initial "learning window," the policy
  records all destinations as known. After the window closes, any new
  destination is flagged.

**Feasibility with vnet policy API:** Fully supported. A stateful policy
tracks seen destinations in `on_dns_query()` and `on_tcp_connect()`:

```rust
struct FirstSeenTracker {
    seen_domains: RwLock<HashSet<String>>,
    seen_ips: RwLock<HashSet<Ipv4Addr>>,
    learning_until: Option<Instant>,
}

impl EgressPolicy for FirstSeenTracker {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
        let mut seen = self.seen_domains.write().unwrap();
        if seen.insert(ctx.domain.clone()) {
            // First time seeing this domain
            if self.is_learning() {
                PolicyAction::Allow { reason: Some(PolicyReason::Custom(
                    format!("learning: first seen {}", ctx.domain).into()
                ))}
            } else {
                PolicyAction::Observe { reason: PolicyReason::Custom(
                    format!("novel domain: {}", ctx.domain).into()
                )}
            }
        } else {
            PolicyAction::Allow { reason: None }
        }
    }

    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyAction {
        let mut seen = self.seen_ips.write().unwrap();
        if seen.insert(ctx.dst_ip) {
            if self.is_learning() {
                PolicyAction::Allow { reason: None }
            } else {
                PolicyAction::Observe { reason: PolicyReason::Custom(
                    format!("novel IP: {}:{}", ctx.dst_ip, ctx.dst_port).into()
                )}
            }
        } else {
            PolicyAction::Allow { reason: None }
        }
    }
}
```

No additional API surface needed. All context fields required for
tracking are already present in `DnsQueryContext` and `TcpConnectContext`.

## Feasibility Summary

| Detector | API coverage | Notes |
|----------|-------------|-------|
| **Entropy** | Full | `DnsQueryContext.domain` + `label_lengths` |
| **Byte ratio** | Full | `TcpFlowContext.bytes_tx` / `bytes_rx` |
| **Flow duration** | Full | `TcpFlowContext.duration` + `is_close` |
| **Beacon/heartbeat** | Partial | Needs stateful inter-report timing; report interval not policy-configurable |
| **First seen** | Full | Stateful policy with `HashSet` in `on_dns_query` / `on_tcp_connect` |

All five detectors are implementable with the current vnet policy API.
The beacon detector has the weakest fit because periodicity detection
requires per-report timestamps accumulated over multiple callbacks,
and the report interval is not policy-controlled. Future API enhancement:
configurable `on_tcp_flow` report interval or a lightweight per-packet
metadata callback.
