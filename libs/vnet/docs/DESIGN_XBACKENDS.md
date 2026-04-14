# motlie-vnet: Cross-Backend Egress Engine Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Initial cross-backend design: split `motlie-vnet` into a reusable slirp/policy/flow core plus hypervisor-specific packet adapters so CH stays stable while Vz and future backends can add policy-capable egress without breaking Linux abstractions |

## Purpose

This document extends [`DESIGN.md`](./DESIGN.md) with the architecture needed
to support multiple hypervisor backends without weakening the existing Linux
Cloud Hypervisor path.

The key requirement is:

- preserve the current `motlie-vnet` value proposition:
  - all userspace
  - no persistent host network configuration changes
  - no elevated privileges
  - runtime state is ephemeral and disappears with the host process
- preserve the `#133` policy-engine direction:
  - DNS query/response observability
  - TCP connect observability
  - deny/allow policy decisions with fail-closed behavior
  - domain-to-IP correlation and flow tracking
- add a path for Vz and future backends without turning `motlie-vnet` into a
  grab bag of unrelated hypervisor-specific code

This means the project should split into:

1. a reusable egress engine core
2. thin backend/platform adapters that move packets between the guest NIC
   frontend and the reusable core

## Design Constraints

These constraints come from the current `motlie-vnet` contract and remain
non-negotiable for the cross-backend design.

### Host Impact

The egress solution must preserve:

- no persistent host network configuration changes
- no TAP device creation as a product requirement
- no route table or firewall/pf edits as a product requirement
- no host interface provisioning as a product requirement
- all userspace operation
- host-side runtime state is ephemeral and tied to backend lifetime

Ephemeral runtime state is allowed:

- Unix sockets
- eventfds or pipes
- host TCP/UDP sockets opened by the process
- temporary files / observability artifacts

Persistent or operator-managed host networking changes are not allowed as the
default product path.

### Policy Parity

The reusable engine must be able to satisfy the policy/observability
requirements currently being designed for `motlie-vnet` in `#133`:

- DNS query inspection before host egress
- DNS response inspection and correlation
- TCP connect inspection before host egress
- policy callbacks that can allow or deny
- response forging for denied DNS/TCP outcomes
- structured events and flow observability

If a backend cannot feed guest traffic through that reusable engine, it cannot
claim policy parity.

## Architectural Split

### Layer 1: Reusable Egress Engine

This is the portable heart of `motlie-vnet`.

Responsibilities:

- own one libslirp instance per guest
- own policy evaluation and event emission
- parse and inspect guest TX packets before host egress
- parse and inspect guest RX packets when needed for correlation/observability
- maintain DNS response reverse maps
- maintain TCP flow/connect state
- generate forged guest-facing responses when policy denies a request
- expose backend-neutral observability

Non-responsibilities:

- vhost-user protocol
- Cloud Hypervisor specifics
- Apple Virtualization specifics
- TAP setup
- QEMU command-line concerns
- Firecracker jailer concerns

### Layer 2: Packet Adapters

This layer is backend-specific.

Responsibilities:

- receive raw L2 guest packets from the hypervisor/frontend
- hand guest TX frames into the reusable engine
- take guest RX frames from the reusable engine and return them to the guest
- surface link-up/link-down/readiness/shutdown state
- configure the guest-facing NIC frontend transport for that hypervisor

Non-responsibilities:

- domain policy decisions
- DNS/TCP parsing policy logic
- flow correlation
- libslirp lifecycle beyond feeding frames to/from the engine

### Layer 3: VMM Runtime Composition

This layer lives above `libs/vnet`, typically in `libs/vmm`.

Responsibilities:

- select which adapter/backend is used for a given hypervisor
- surface product-level egress capabilities:
  - outbound internet available
  - policy-capable egress available
  - no persistent host config drift
- integrate observability into VM lifecycle and harness reporting

## Proposed Modules

The exact names can change, but the split should look roughly like this:

```text
libs/vnet/src/
├── lib.rs
├── config.rs
├── error.rs
├── engine/
│   ├── mod.rs
│   ├── slirp_engine.rs
│   ├── packet.rs
│   ├── flow.rs
│   ├── dns.rs
│   ├── forge.rs
│   └── observe.rs
├── policy/
│   ├── mod.rs
│   ├── traits.rs
│   ├── chain.rs
│   ├── category.rs
│   └── no_policy.rs
└── adapter/
    ├── mod.rs
    ├── ch_vhost_user.rs
    ├── vz.rs
    ├── qemu.rs
    └── firecracker.rs
```

The important point is structural:

- `engine/` and `policy/` are portable
- `adapter/` is where backend coupling belongs

## Core Traits And API Boundaries

### `EgressPolicy`

This stays in the reusable core and remains backend-neutral.

```rust
pub trait EgressPolicy: Send + Sync + 'static {
    fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyDecision;
    fn on_dns_response(&self, ctx: &DnsResponseContext) -> PolicyDecision;
    fn on_tcp_connect(&self, ctx: &TcpConnectContext) -> PolicyDecision;
}
```

Responsibilities:

- evaluate guest intent
- return allow/deny plus metadata
- remain independent of hypervisor transport

### `PacketEgressEngine`

This is the reusable engine front door.

```rust
pub trait PacketEgressEngine: Send + 'static {
    fn ingest_guest_tx(&mut self, frame: &[u8]) -> Result<EngineIngress, VnetError>;
    fn poll_guest_rx(&mut self) -> Result<Vec<GuestRxFrame>, VnetError>;
    fn observability(&self) -> EgressObservability;
    fn shutdown(&mut self) -> Result<(), VnetError>;
}
```

Suggested semantics:

- `ingest_guest_tx(...)`
  - receives one guest-originated Ethernet frame
  - performs policy checks before host egress
  - may:
    - forward into libslirp
    - deny and queue a forged guest RX response
    - emit events only
- `poll_guest_rx(...)`
  - drains any guest-bound frames produced by:
    - libslirp
    - deny-forging logic
    - DHCP/DNS internal services

### `PacketAdapter`

This is the boundary between a specific hypervisor frontend and the reusable
engine.

```rust
pub trait PacketAdapter: Send + 'static {
    type Config;

    fn new(config: Self::Config) -> Result<Self, VnetError>
    where
        Self: Sized;

    fn run(
        self,
        engine: impl PacketEgressEngine,
        lifecycle: AdapterLifecycleSink,
    ) -> Result<AdapterRunReport, VnetError>;
}
```

Responsibilities:

- translate hypervisor/device events into raw packet reads/writes
- own backend-specific sockets/fds/file handles
- report adapter lifecycle state

Non-responsibilities:

- policy
- flow classification
- DNS/TCP semantics
- observability meaning beyond local transport status

### Apple Vz Adapter Note

The Vz parity path must use
`VZFileHandleNetworkDeviceAttachment`, not Apple NAT.

That distinction matters because:

- `VZNATNetworkDeviceAttachment`
  - is useful for bootstrap/debug internet access
  - preserves the no-persistent-host-config property
  - does not expose packets to the Rust-owned policy engine
- `VZFileHandleNetworkDeviceAttachment`
  - is the concrete Vz API that can plausibly feed guest TX/RX packets into the
    reusable `PacketEgressEngine`
  - is therefore the only Vz path that can claim `#133`-style policy parity

### `AdapterLifecycleSink`

This should be a small shared reporting surface for adapters.

```rust
pub trait AdapterLifecycleSink: Send + Sync + 'static {
    fn on_ready(&self);
    fn on_link_up(&self);
    fn on_link_down(&self, reason: &str);
    fn on_error(&self, stage: &'static str, error: &str);
}
```

The goal is to normalize transport lifecycle without contaminating the policy
engine with backend details.

## Backend Responsibilities

### Cloud Hypervisor Adapter

Module:

- `adapter::ch_vhost_user`

Responsibilities:

- preserve current Linux behavior exactly
- own the vhost-user backend/socket logic
- move packets between virtqueues and `PacketEgressEngine`
- preserve existing CH API and observability during refactor

Non-goals:

- no behavioral redesign while extracting the core
- no new policy semantics beyond what `#133` introduces

### Vz Adapter

Module:

- `adapter::vz`

Responsibilities:

- move guest NIC packets between Apple Vz networking attachment APIs and the
  reusable engine
- preserve the same egress policy and observability semantics as CH
- preserve the no-persistent-host-config-change requirement

Critical constraint:

- Apple backend-native NAT by itself is not sufficient for parity because Rust
  cannot inspect or deny DNS/TCP intent decisions inside Apple’s NAT path

Implication:

- `adapter::vz` must provide a packet path into the reusable engine
- if Vz cannot provide that path, policy parity is blocked

### QEMU Adapter

Module:

- `adapter::qemu`

Responsibilities:

- feed guest NIC traffic into the same reusable engine through a QEMU-compatible
  packet transport

Notes:

- QEMU already demonstrates the viability of slirp/user-mode networking as a
  concept
- the adapter work is transport-specific, not policy-specific

### Firecracker Adapter

Module:

- `adapter::firecracker`

Responsibilities:

- bridge Firecracker guest NIC traffic into the same reusable engine

Notes:

- Firecracker is more constrained than CH/QEMU and may require a different
  transport realization
- that does not invalidate the reusable engine split; it only affects adapter
  feasibility

## Public API Evolution

The public API should evolve conservatively so Linux callers do not see a
breaking abstraction shift before the split is proven.

### Phase 0 API

Keep the current CH-oriented API intact:

```rust
VnetConfig::builder()
    .socket_path(...)
    .guest_ipv4(...)
    .host_ipv4(...)
    .netmask(...)
    .dns_ipv4(...)
    .mac(...)
    .build()?;

VnetBackend::new(config).start()?;
```

### Phase 1 API

Keep the same public entrypoint, but reimplement internally in terms of:

- CH adapter
- reusable engine

No public behavior change.

### Phase 2 API

After CH is stable on the new internals, introduce a backend-neutral config
layer above adapters.

Suggested shape:

```rust
pub enum VnetBackendKind {
    CloudHypervisorVhostUser,
    VzPacketAdapter,
    QemuPacketAdapter,
    FirecrackerPacketAdapter,
}

pub struct VnetInstanceConfig<P> {
    pub backend: VnetBackendKind,
    pub adapter: AdapterConfig,
    pub engine: EngineConfig<P>,
}
```

Important:

- `VnetConfig` for CH can remain as a compatibility builder
- generic/backend-neutral config should be added, not forced onto old callers

## Recommended Project Phasing

### Phase 1: Lock Current Linux Behavior

- add or tighten compatibility tests around current CH behavior
- add policy-engine tests for the `#133` semantics
- define "no regression" as:
  - same startup/shutdown behavior
  - same guest DHCP/DNS/egress behavior
  - same socket lifecycle
  - same host-impact properties

Goal:

- prevent refactor drift before the split starts

### Phase 2: Extract Internal Engine Modules

- move slirp, policy, flow, DNS, and forging logic behind internal engine
  boundaries
- keep the current CH public API unchanged

Goal:

- create the reusable core without forcing any new abstraction on Linux callers

### Phase 3: Introduce Adapter Trait Internally

- implement the current CH path as `adapter::ch_vhost_user`
- keep old `VnetBackend::start()` behavior and tests intact

Goal:

- prove the split on the already working Linux backend first

### Phase 4: Expose Backend-Neutral Observability

- add backend-neutral observability structs for:
  - egress ready/not ready
  - policy-capable/not policy-capable
  - flow event counts
  - DNS/TCP deny counts
- keep CH-specific observability as a compatibility view if needed

Goal:

- make `libs/vmm` depend on outcomes rather than CH transport specifics

### Phase 5: Spike Vz Adapter Feasibility

- prototype a Vz packet adapter without altering the stable CH path
- verify:
  - packet ingress from guest
  - packet egress back to guest
  - no persistent host config drift
  - preserved policy visibility

Goal:

- de-risk Vz before committing to a large backend merge

### Phase 6: Add Generic Config/API Surface

- only after CH and the Vz spike are both proven
- add backend-neutral config entrypoints for multi-backend use

Goal:

- avoid premature public abstractions that end up distorted by one backend

## What Must Not Happen

- do not add Apple NAT as `libs/vnet/nat`
- do not make `motlie-vnet` mean "whatever networking feature a hypervisor
  happens to provide"
- do not move DNS/TCP policy logic into `backend::vz`
- do not weaken `#133` policy requirements to make Vz easier
- do not break the Linux CH path while chasing adapter generality

## Acceptance Criteria For The Split

The split is successful when all of these are true:

- CH behavior is unchanged from the caller perspective
- the reusable engine owns the full `#133` policy surface
- adapters are thin and transport-specific
- `libs/vmm` can ask for policy-capable egress without naming a specific
  transport
- Vz support either:
  - feeds packets into the reusable engine and gets policy parity, or
  - is declared non-parity and remains outside the policy-capable path

## Crate-Boundary Decision

`motlie-vnet` should evolve into:

- the reusable userspace egress engine crate

It should not evolve into:

- a generic wrapper over any native hypervisor networking feature

That distinction preserves the Linux core, makes the policy engine reusable,
and gives Vz/QEMU/Firecracker a path to adopt the same behavior through thin
adapters instead of copy-pasted policy stacks.
