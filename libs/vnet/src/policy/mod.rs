//! Network policy detection primitives organized by connection phase.
//!
//! A network connection is a timeline with three detection phases:
//!
//! ## Phase 1: DNS — The "Intent" Phase
//!
//! The guest asks "where is this?" before any connection. Best place to
//! catch stealthy smuggling before a byte of real data is exchanged.
//!
//! - [`entropy`] — Shannon entropy of DNS labels. Detects encoded payloads
//!   in subdomain strings (base64, hex). Uses `DnsQueryContext.domain` and
//!   `label_lengths`.
//! - [`domain`] — Base domain extraction and suffix matching. Enables
//!   rate tracking per base domain and category classification.
//!
//! Also applicable at this phase (stateful, not separate modules):
//! - **Request frequency** — rate counter per base domain via
//!   `DnsQueryContext.domain` + `timestamp` in a `RwLock<HashMap>`.
//! - **Record type profiling** — flag rare types (TXT, NULL) via
//!   `DnsQueryContext.query_type`.
//! - **Domain aging** — external WHOIS lookup keyed by `domain` (async,
//!   return Observe and resolve out-of-band).
//!
//! ## Phase 2: Pre-Connection / Initial Handshake
//!
//! TCP SYN and TLS negotiation. Looking at identity and parameters.
//!
//! Applicable detectors (stateful, using `TcpConnectContext` and first
//! `TcpFlowContext` report):
//! - **SNI / DNS mismatch** — compare `TcpFlowContext.sni` with
//!   `TcpFlowContext.domain` to detect domain fronting.
//! - **Protocol validation** — check first TX bytes for expected TLS
//!   record header vs raw data on port 443.
//! - **Geo-fencing / rare IP** — compare `TcpConnectContext.dst_ip`
//!   against historical baselines or GeoIP databases.
//! - **JA3 fingerprinting** (future) — hash TLS ClientHello parameters
//!   to identify malicious tools. Requires raw ClientHello bytes in
//!   `TcpFlowContext` (not yet exposed).
//!
//! ## Phase 3: Flow — The "Established" Session
//!
//! Connection is live, data is moving. Analyze shape and rhythm.
//!
//! - Future: [`ratio`] — outbound/inbound byte ratio for exfiltration.
//!   Uses `TcpFlowContext.bytes_tx / bytes_rx`.
//! - Future: [`duration`] — long-lived low-throughput flow detection.
//!   Uses `TcpFlowContext.duration` and bytes/sec rate.
//! - Future: [`beacon`] — periodic heartbeat detection via inter-report
//!   timing analysis. Uses `TcpFlowContext.timestamp` across reports.
//! - Future: [`first_seen`] — novel destination tracking. Uses
//!   `DnsQueryContext.domain` and `TcpConnectContext.dst_ip`.
//!   Spans Phase 1 + Phase 2.

pub mod entropy;
pub mod domain;
