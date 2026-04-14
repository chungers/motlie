# motlie-vmm: Virtualization.framework Backend Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Align the Vz backend phases with the cross-backend slice order, formalize the hardened SSH auto-provision checks as parity acceptance tests, and replace the stale VFS-parity-definition ambiguity with an explicit `v1.15` feasibility gate |
| 2026-04-14 | @codex-vz | Add `DESIGN_GUEST_IMAGE.md` as the owning prerequisite for the Apple Vz guest image pipeline: define the boot artifacts, guest image build options, aarch64 kernel constraints, packaging direction, and the shared-guest-contract vs hypervisor-specific-packaging split |
| 2026-04-13 | @codex-vz | Address PR 163 review findings: declare the macOS 12 floor, add the helper entitlement/signing and build/discovery story, define the Rust↔Swift config/control contracts, harden readiness, describe vsock/cloud-init delivery, and tighten several factual details |
| 2026-04-12 | @codex-vz | Initial design for a macOS `backend::vz` that satisfies the current `Runtime` / `VmHandle` contracts with Apple `Virtualization.framework`, recommends a Swift helper process for the first slice, and defines the portability gaps that need small API cleanup in `libs/vmm` |

## Goal

Add a real `BackendKind::Vz` path for macOS so `motlie-vmm` can boot and
manage Linux guests through Apple `Virtualization.framework` while preserving
the current top-level lifecycle:

- `prepare()`
- `boot()`
- `VmHandle::ready(...)`
- `VmHandle::exec(...)`
- `VmHandle::open_pty(...)`
- `VmHandle::shutdown()`
- `VmHandle::observability()`

The design target is not feature parity with the Motlie-backed Cloud
Hypervisor stack on day one. The first Vz slice should support the same
orchestrator contracts with a narrower backend capability set:

- Linux guest boot on macOS
- SSH control plane over virtio-vsock
- host-visible serial log capture
- host directory sharing for guest mounts
- policy-capable outbound guest networking
- deterministic shutdown and exit observation

## Current Local Constraints

The current code already gives us the right layering, but it still carries a
few CH-shaped assumptions that must be relaxed for Vz:

- [`libs/vmm/src/backend/mod.rs`](../src/backend/mod.rs)
  - `BackendHandle` only has `ChShell`
- [`libs/vmm/src/runtime.rs`](../src/runtime.rs)
  - `HypervisorBacking::AppleVirtualization` exists, but returns
    `UnsupportedHypervisor`
- [`libs/vmm/src/backend/vz/mod.rs`](../src/backend/vz/mod.rs)
  - placeholder only
- [`libs/vmm/src/orchestrator.rs`](../src/orchestrator.rs)
  - `ready()` unconditionally waits for `runtime_paths.api_socket`
  - run-artifact reporting treats `api_socket` as a required run artifact
- [`libs/vmm/src/spec.rs`](../src/spec.rs)
  - `GuestRuntimePaths` is generic enough for Vz, but the semantic meaning of
    `api_socket` is CH-specific today

The important design consequence is:

`VmHandle` and the orchestrator API are already reusable, but readiness and
observability need one small abstraction step so non-CH backends are not forced
to fake a Cloud Hypervisor API socket.

## External Capability Findings

As of 2026-04-12, the relevant Apple and Rust surfaces look like this:

- Apple `Virtualization.framework` directly supports the primitives we need for
  a Linux guest:
  - `VZLinuxBootLoader` on macOS 11+, but it must be paired with
    `VZGenericPlatformConfiguration`
  - `VZVirtioFileSystemDeviceConfiguration` on macOS 12+
  - `VZVirtioSocketDeviceConfiguration` on macOS 11+
  - `VZNATNetworkDeviceAttachment` on macOS 11+
  - `VZFileHandleSerialPortAttachment` on macOS 11+
  - `VZVirtualMachineConfiguration.platform` on macOS 12+
- Apple also exposes higher-end options we can defer:
  - `VZBridgedNetworkDeviceAttachment` with the
    `com.apple.vm.networking` entitlement
  - `VZVmnetNetworkDeviceAttachment` on macOS 26+ only
  - graphics, ballooning, Rosetta directory sharing, NBD, USB
- The older `virtualization-rs` crate is not a good primary choice:
  - docs.rs shows version `0.1.2`
  - docs.rs does not list `aarch64-apple-darwin`
  - it uses the older `objc` crate bindings
- The current Rust binding surface is `objc2-virtualization`:
  - docs.rs shows version `0.3.2`
  - it exposes the needed Vz classes and protocols
  - docs.rs also shows Virtualization objects such as `VZNetworkDevice` as
    `!Send` / `!Sync`

The `!Send` / `!Sync` point matters because the current orchestrator stores
backend state inside `VmHandle`, and those handles are expected to be usable
from ordinary async Rust code.

## Platform Floor

The effective phase-1 deployment floor is macOS 12.0 (Monterey).

Reasoning:

- Linux boot, vsock, serial, and NAT all exist on macOS 11+
- phase 1 also requires VirtioFS host directory sharing
- the required VirtioFS/share APIs are macOS 12+:
  - `VZVirtioFileSystemDeviceConfiguration`
  - `VZSharedDirectory`
  - `VZMultipleDirectoryShare`
  - `VZVirtualMachineConfiguration.platform`

This needs to be explicit in both implementation and build settings:

- `vz-runner` should compile with a minimum deployment target of macOS 12.0
- Rust `backend::vz` code should be `#[cfg(target_os = "macos")]`
- any future Rust-native Vz bindings should also assume a macOS 12 floor unless
  the design is split into a reduced non-VirtioFS path

Apple Silicon is the primary target for phase 1, but the helper binary should
be built as a universal macOS binary where practical so Intel developer hosts
remain usable.

## Decision

Phase 1 should implement Vz through a small helper process written in Swift:

- new helper binary: `vz-runner`
- Rust `backend::vz` spawns and supervises that helper
- helper owns all `Virtualization.framework` objects and delegate callbacks
- helper emits:
  - a readiness sentinel file
  - a serial log file
  - a small control socket for shutdown and status

Rust should not directly own `VZVirtualMachine` objects in phase 1.

### Why Swift First

This is the lowest-risk path for the current codebase:

- it preserves the existing process-oriented `BackendHandle` pattern from
  `backend::ch::shell`
- it avoids mixing Objective-C object lifetime rules into `VmHandle`
- it avoids pushing `!Send` / `!Sync` Vz objects through the async runtime
- it makes delegate-driven state changes easy to model with native Apple APIs
- it keeps a later Rust-native implementation possible behind the same backend
  contract

### Why Not `virtualization-rs`

It looks stale relative to the current Apple API surface and, based on docs.rs,
is not the right Apple Silicon-first foundation for a new macOS backend.

### Why Not Pure `objc2-virtualization` First

It is viable long term, but it raises complexity immediately:

- Objective-C ownership and delegate bridging in Rust
- thread-affinity and run-loop concerns
- `!Send` / `!Sync` VM object graphs
- more unsafe surface before we have even proven the backend contract

That is the wrong order for a first backend slice.

## Entitlements And Signing

`Virtualization.framework` changes the deployment story relative to the current
pure-Rust CH shell backend.

### Required Entitlements

Phase 1 `vz-runner` needs:

- `com.apple.security.virtualization`

Deferred/network-expansion work would additionally need:

- `com.apple.vm.networking`
  - only for bridged networking, not for the phase-1 NAT path

### Which Binary Carries The Entitlement

The entitlement belongs on the process that directly instantiates
`Virtualization.framework` objects.

For the helper-process design, that means:

- `vz-runner` must be signed with the virtualization entitlement
- the Rust `motlie-vmm` binary does not need that entitlement merely to spawn
  `vz-runner`
- if the helper is ever embedded into a single macOS app bundle or replaced by
  an in-process backend, the entitlement story must be re-evaluated for the new
  executable boundary

### Development Signing

The design should assume:

- local development uses either ad-hoc signing or a development certificate
- `vz-runner` still needs an entitlements plist during local development
- "unsigned helper" is not a supported assumption for this design; the
  implementation should require explicit signing rather than depending on
  environment-specific SIP behavior

### Distribution Signing

For distribution:

- `vz-runner` must ship signed with the virtualization entitlement
- bridged-network builds would require an additional signing profile carrying
  `com.apple.vm.networking`
- the design should treat helper signing as part of the build artifact, not as
  a post-install manual step

### Repository Artifacts

Phase 2 should add a checked-in entitlements file such as:

- `libs/vmm/vz/vz-runner.entitlements`

with at least:

```xml
<key>com.apple.security.virtualization</key>
<true/>
```

This makes the entitlement requirement visible in version control instead of
burying it in local Xcode state.

## Helper Discovery And Build Integration

The design needs one explicit rule for how Rust discovers the Swift helper and
how the workspace builds it.

### Runtime Discovery

Phase 1 should resolve `vz-runner` in this order:

1. `MOTLIE_VZ_RUNNER=/abs/path/to/vz-runner`
2. a sibling binary next to the Rust executable
3. a checked repository build output under a known relative path
4. `$PATH` as a last-resort developer convenience

That order keeps distribution deterministic while still allowing local override.

The backend should report the resolved path in observability and launch logs.

### Build Integration

Phase 1 should define one supported build path:

- `cargo build -p motlie-vmm` builds Rust only
- a dedicated repo script or Make target builds and signs `vz-runner`
- Rust runtime code fails fast with a clear error if `AppleVirtualization` is
  selected and the helper is missing

This keeps Swift/Xcode toolchain requirements out of normal Linux CI and
non-macOS Rust builds.

Recommended repo-owned build entrypoint:

- `libs/vmm/vz/build-vz-runner.sh`

Responsibilities:

- invoke `xcodebuild` or `swift build`
- set the minimum deployment target to macOS 12.0
- build a universal binary where feasible
- sign the resulting binary with the checked-in entitlements plist
- place the final artifact in a predictable repo-local output path

### CI

The design should explicitly assume:

- Linux CI does not build or test `vz-runner`
- macOS CI builds `vz-runner` only in dedicated jobs
- integration tests needing live virtualization run only on macOS runners with
  the required entitlement/signing setup

## Backend Shape

### New Backend Types

`backend/mod.rs` should grow:

```rust
pub enum BackendHandle {
    ChShell(ChShellHandle),
    Vz(VzHandle),
}
```

`backend/vz/mod.rs` should define:

```rust
pub struct VzHandle {
    pub pid: Option<u32>,
    pub control_socket: std::path::PathBuf,
    pub readiness_path: std::path::PathBuf,
    pub serial_log_path: std::path::PathBuf,
    pub child: std::sync::Mutex<Option<std::process::Child>>,
}

pub struct VzBackend;
```

The handle should intentionally look like `ChShellHandle`: a process plus a
small set of filesystem/control artifacts that Rust can poll and clean up.

### Backend Capabilities

`VzBackend::capabilities()` should be approximately:

```rust
VmBackendCapabilities {
    same_process_vmm: false,
    supports_api_socket: false,
    supports_event_monitor: false,
    supports_fd_handoff: false,
    supports_memfd_boot_artifacts: false,
    supports_guest_metrics: false,
}
```

`same_process_vmm` stays `false` in phase 1 because the first implementation is
helper-process based.

## Runtime Contract Changes

### 1. Replace hardcoded API-socket readiness with backend readiness

Current `VmHandle::ready(...)` assumes CH:

- wait for `runtime_paths.api_socket`
- then guestfs, SSH bridge, exec

Vz needs:

- wait for backend readiness sentinel
- then guestfs, SSH bridge, exec

Recommended change:

```rust
impl BackendHandle {
    pub fn readiness_probe(&self) -> BackendReadinessProbe;
}

pub enum BackendReadinessProbe {
    Path(std::path::PathBuf),
    None,
}
```

Then `VmHandle::ready(...)` waits on `backend_handle.readiness_probe()` instead
of `runtime_paths.api_socket`.

For CH, the probe remains `runtime_paths.api_socket`.

For Vz, the probe is `runtime_paths.launch_dir.join("vz-ready")`.

Behavioral compatibility requirement:

- CH remains `Path(runtime_paths.api_socket)`
- existing CH tests should continue to pass without semantic changes
- the abstraction must stay as a cheap enum/path lookup, not a new async layer

### 2. Make observability backend-aware

`VmObservability` should continue exposing `runtime_paths.api_socket`, but the
run artifact should stop claiming it is always required.

Recommended change:

- keep `GuestRuntimePaths.api_socket` for CH compatibility
- add a backend-specific requiredness rule in `VmHandle::run_artifacts(...)`
- add a new optional artifact kind:
  - `BackendReady`
  - or `BackendControlSocket`

This preserves existing CH output while making Vz honest.

### 3. Keep `VmHandle` lifecycle unchanged

The orchestrator should not gain a separate Vz-specific VM type. The whole
point of the backend is to satisfy the existing shape:

- `boot()` returns `VmHandle`
- `VmHandle::shutdown()` delegates to `Runtime.hypervisor.shutdown(...)`
- `VmHandle::exec()` and `open_pty()` continue to ride the existing SSH bridge

## Guest Input Mapping

### Boot Artifacts

Current `GuestSpec.boot` maps cleanly:

- `boot.kernel` -> `VZLinuxBootLoader(kernelURL:)`
- `boot.initramfs` -> `initialRamdiskURL`
- `boot.cmdline` -> `commandLine`
- `boot.firmware`
  - not used for the normal Linux path in phase 1
  - if present, backend should reject it with a clear unsupported error that
    explains EFI boot on Vz requires a different configuration path than
    `VZLinuxBootLoader`, not merely that the field is temporarily unsupported

### CPU / Memory

Current `GuestResources` maps directly:

- `boot_vcpus` -> `VZVirtualMachineConfiguration.cpuCount`
- `memory_mib` -> `memorySize = memory_mib * 1024 * 1024`
- `max_vcpus`
  - no phase-1 use
  - ignore or reject until hotplug / topology policy exists

The helper should validate CPU and memory against the runtime-reported Vz
limits before calling `validate()`:

- `minimumAllowedCPUCount ..= maximumAllowedCPUCount`
- `minimumAllowedMemorySize ..= maximumAllowedMemorySize`

Those validation failures should come back to Rust as structured launch errors,
not as opaque helper crashes.

### Storage

Current `GuestStorage.overlay_size` does not map directly to Vz. Vz wants a
block device attachment backed by a disk image or block device, not a CH-style
overlay-size string.

Phase 1 policy:

- require the caller to provide a bootable writable disk image in the guest
  artifact set used by the example/harness
- treat `overlay_size` as unused by Vz for now
- document that portable storage modeling still needs a follow-up type split:
  - CH overlay policy
  - Vz disk-image attachment policy

This is the largest guest-shape mismatch in the current API.

### Mounts

Current `GuestMountSpec` can map to VirtioFS:

- each mount tag -> `VZVirtioFileSystemDeviceConfiguration.tag`
- host path -> `VZSharedDirectory`
- one mount per device in phase 1

Inside the guest, the existing cloud-init `mounts.yaml` flow can stay in place,
but Vz still needs an explicit cloud-init delivery path.

Phase 1 delivery mechanism:

- create a small `cidata` disk image from the already-rendered `meta-data`,
  `user-data`, and `mounts.yaml`
- attach that disk as a read-only block device so cloud-init can consume the
  normal NoCloud datasource during boot
- keep VirtioFS responsible only for host-directory sharing, not for delivering
  the cloud-init datasource itself

The backend therefore has two separate responsibilities:

- ensure host shares exist with matching tags before boot
- ensure the NoCloud datasource is attached as a separate boot-time artifact

This means `FilesystemBacking::HypervisorManaged` is the correct default Vz path
for shared directories. We do not need `backend::motlie::vfs` for the first Vz
slice.

### Networking

The current `NetworkModes` model is CH/Motlie-oriented. Vz phase 1 should not
try to emulate every existing mode.

Revised decision:

- Apple NAT may exist as a helper/bootstrap mode for bring-up and local
  debugging, but it is not sufficient for backend parity
- the parity path must preserve the policy and observability requirements being
  designed for `motlie-vnet` in issue/PR `#133`
- therefore Vz networking should be designed around a policy-capable egress
  backend, not around `VZNATNetworkDeviceAttachment` alone

Required parity properties for outbound egress:

- DNS query/response observability
- TCP connect observability
- policy callbacks at the Rust packet/flow layer
- domain-to-IP correlation for policy decisions
- deny behavior that can fail closed before host egress occurs

Recommended Vz mapping:

- `AdminNetMode::None + EgressNetMode::None`
  - boot without a network device
- `AdminNetMode::None + EgressNetMode::VhostUser`
  - do not map directly to Apple NAT for the parity implementation
  - instead, represent "policy-capable outbound egress required"
  - the concrete Vz realization still needs design work:
    - adapt `motlie-vnet` or an equivalent policy-capable egress engine to a
      Vz-compatible frontend path
    - or explicitly introduce a separate Vz-specific policy-capable egress
      layer with the same semantics as `motlie-vnet`
- all other combinations
  - reject as unsupported for Vz phase 1

Reasoning:

- `motlie-vnet` is the reusable policy-capable egress subsystem in the current
  crate set; Apple NAT is a backend-native capability, not a reusable service
- simple NAT is close to "guest has outbound network access", but it cannot by
  itself satisfy the observability/policy requirements from `#133`
- the intended phase-1 property is no persistent host network configuration
  changes:
  - no TAP device creation
  - no route or pf rule changes
  - no manual interface provisioning
  - only helper/OS-owned ephemeral runtime networking state while the VM runs
- preserving that host-impact property remains a requirement even if the final
  Vz parity path is not Apple NAT
- bridged and vmnet custom topologies can come later, but they should be added
  as explicit Vz network modes, not hidden behind CH vocabulary

This implies `validate_network_modes(...)` must eventually become
backend-sensitive, or Vz must add its own validation during boot.

### Vsock

Current control-plane design already fits Vz very well:

- `backend::motlie::ssh_proxy` binds a host-side UDS derived from
  `runtime_paths.vsock_socket`
- the guest is expected to connect out to host port `2222`

Vz phase 1 should configure:

- `VZVirtioSocketDeviceConfiguration`
- a host listener for the guest SSH bridge port

The helper process must accept guest vsock connections and bridge them onto the
same host UDS path that the existing Rust SSH proxy expects.

That keeps `VmHandle::exec(...)` and `VmHandle::open_pty(...)` unchanged.

Bridge shape in phase 1:

- Rust still owns the SSH proxy logic and its Unix domain listener path
- `vz-runner` owns the Vz host-side virtio-socket listener
- for each guest-initiated connection to the SSH bridge port, `vz-runner`
  opens a fresh Unix stream connection to the Rust-owned UDS
- it then performs a byte-for-byte duplex copy until EOF on either side
- no additional framing or protocol translation is introduced

Lifecycle requirements:

- multiple sequential or concurrent guest bridge connections must be supported
- bridge tasks must be cancelled when the VM exits or helper shutdown begins
- helper-side bridge errors must be reflected in control-socket status so Rust
  can fail readiness and emit actionable diagnostics

## `vz-runner` Process Model

The helper should take a fully materialized launch description from Rust and
own the VM until exit.

### Rust Side

`VzBackend::boot(prepared)` should:

1. create launch/runtime directories
2. materialize a versioned JSON config file in `runtime_paths.launch_dir`
3. pre-create files and sockets needed for logs/control
4. spawn `vz-runner`
5. return `BackendHandle::Vz(VzHandle { ... })`

### Helper Side

`vz-runner` should:

1. decode the JSON config
2. build `VZVirtualMachineConfiguration`
3. validate configuration before launch
4. construct `VZVirtualMachine`
5. start the VM
6. wait until:
   - the VM reports `running`
   - the control socket is accepting requests
   - the vsock bridge listener is accepting guest connections
7. mark readiness by creating `vz-ready`
8. service a tiny control socket until the VM exits or shutdown is requested

### `vz-config.json` Contract

The Rust->Swift config file should be a stable versioned schema, not an ad hoc
bag of launch arguments.

Recommended top-level shape:

```json
{
  "schema_version": 1,
  "guest_id": "alice",
  "platform": {
    "minimum_macos": "12.0"
  },
  "boot": {
    "kernel": "/abs/path/Image",
    "initramfs": "/abs/path/initrd.img",
    "cmdline": "console=hvc0"
  },
  "resources": {
    "cpu_count": 2,
    "memory_bytes": 536870912
  },
  "storage": {
    "root_disk": "/abs/path/root.img",
    "cloud_init_disk": "/abs/path/cidata.iso"
  },
  "mounts": [
    {
      "tag": "alice-home",
      "host_path": "/abs/path/alice-home",
      "read_only": false
    }
  ],
  "network": {
    "mode": "engine"
  },
  "vsock": {
    "ssh_port": 2222,
    "bridge_uds_path": "/abs/path/motlie-vmm-v14-alice.vsock_2222"
  },
  "artifacts": {
    "serial_log_path": "/abs/path/serial.log",
    "launch_log_path": "/abs/path/launch.log",
    "readiness_path": "/abs/path/vz-ready",
    "control_socket_path": "/abs/path/vz-control.sock"
  }
}
```

Field-level notes:

- `schema_version` must be mandatory so Rust and Swift can reject mismatched
  helper/runtime revisions cleanly
- all file paths should be absolute before handoff to Swift
- the config should carry only already-materialized paths, not unresolved
  repository-relative guesses
- `network.mode` must distinguish:
  - `nat`
    - Apple `VZNATNetworkDeviceAttachment`
    - bootstrap/debug only
    - not sufficient for policy parity
  - `engine`
    - policy-capable path using
      `VZFileHandleNetworkDeviceAttachment` plus the Rust-owned
      `PacketEgressEngine`
    - intended parity path for `motlie-vnet`

### Control Socket Protocol

The helper control socket should use line-delimited JSON messages for phase 1.

Rust -> Swift requests:

```json
{"id":1,"command":"status"}
{"id":2,"command":"shutdown"}
```

Swift -> Rust responses/events:

```json
{"id":1,"ok":true,"state":"running","pid":12345,"vsock_bridge_ready":true}
{"id":2,"ok":true,"state":"stopping"}
{"event":"state_changed","state":"stopped","reason":"guest_shutdown"}
{"event":"error","stage":"launch","message":"cpu_count below minimumAllowedCPUCount"}
```

Protocol requirements:

- `status` reports VM lifecycle state, helper pid, and bridge/control readiness
- `shutdown` requests graceful stop and returns an immediate ack
- asynchronous `error` and `state_changed` events are allowed at any time
- unrecoverable helper/bootstrap errors must cross the socket as structured
  events before process exit when possible
- Rust should treat socket disconnect before `stopped` as a launch/shutdown
  failure unless the child has already exited cleanly

### Readiness Robustness

`vz-ready` should not be treated as a blind single-bit success flag.

Phase 1 readiness should be two-phase:

1. `vz-ready` exists and contains:
   - helper pid
   - schema version
   - final transition state used to mark ready
2. a control-socket `status` request succeeds and reports:
   - `state = running`
   - `vsock_bridge_ready = true`

This keeps the simple file-based probe for orchestration while preventing the
obvious stale-sentinel failure mode.

### Suggested On-Disk Artifacts

Under `runtime_paths.launch_dir`:

- `vz-config.json`
- `vz-ready`
- `vz-control.sock`
- `launch.log`
- `serial.log`
- `cloud-init.iso`

Under `runtime_paths.runtime_dir`:

- backend-private transient state if needed

This is intentionally parallel to the CH shell backend’s file-oriented
observability.

### Serial Policy

Phase 1 serial support is capture-only:

- `vz-runner` writes the guest serial stream to `serial.log`
- interactive serial console forwarding is out of scope for the first slice
- future manual-debug tooling can add bidirectional serial later if the harness
  actually needs it

## Shutdown Model

The backend shutdown contract should match the CH model:

1. attempt graceful shutdown over the helper control socket
2. wait for helper exit
3. fall back to `SIGTERM`
4. fall back to `SIGKILL`

`BackendShutdownOutcome` remains sufficient:

- `api_attempted`
  - rename later to something backend-neutral such as `graceful_attempted`
  - until then, Vz can set it to `true` when the control socket path was used
- `forced`
  - remains `Option<&'static str>` with CH-style values such as `"term"` or
    `"kill"`

The helper should implement graceful shutdown by requesting VM stop through
`Virtualization.framework` and then exiting only after the VM reaches a terminal
state.

## Error Model

Add `VzError` under `backend::vz` for:

- config materialization failures
- unsupported guest spec fields
- unsupported network mode combinations
- helper spawn failure
- helper control-socket failure
- early VM exit before readiness
- invalid/missing boot artifacts
- helper/helper-schema version mismatch
- CPU or memory outside Vz-supported bounds
- vsock bridge bootstrap failure
- missing helper binary or missing helper entitlement/signing

Keep those surfaced through `BackendError` and `RuntimeError` so callers still
see them as normal orchestrator failures.

## Implementation Plan

The phase numbering in this document describes the eventual `backend::vz`
implementation inside `libs/vmm`.

It does not replace the earlier proving order in
`libs/vmm/docs/DESIGN_XBACKENDS.md`.

Cross-reference:

| `DESIGN_XBACKENDS.md` stage | Meaning | `DESIGN_VZ.md` relationship |
|---|---|---|
| Stage 0 / `v1.05` | guest image/build proof | prerequisite before Vz backend implementation starts |
| Stage 1 / `v1.15` | managed guestfs proof | prerequisite before Vz backend implementation starts |
| Stage 2 / `v1.25` | policy-capable egress proof | prerequisite before Vz backend implementation starts |
| Stage 3 / CH-safe refactors | reusable infra cleanup | prerequisite or parallel prep for implementation |
| Stage 4 / policy phases | reusable policy semantics | prerequisite for full parity claims |
| Stage 5 / `v1.45` | full VMM Vz slice | maps to the implementation phases below |

### Phase 1: Documented contract cleanup

- add `VzHandle` and `BackendHandle::Vz`
- add backend readiness probe abstraction
- make observability artifact requiredness backend-aware
- add backend-specific network validation hook

### Phase 2: Helper-backed minimal Linux boot

- create `vz-runner`
- support:
  - Linux boot loader
  - one NAT NIC
  - one virtio socket device
  - one serial attachment
  - zero or more virtio-fs shares
- implement graceful shutdown over control socket
- add checked-in entitlements plist and repo-owned helper build script

### Phase 3: Rust integration

- implement `backend::vz::VzBackend`
- wire `HypervisorBacking::AppleVirtualization` to the backend
- add tests for handle kind, shutdown behavior, and readiness probe behavior
- add unit tests for config serialization and control-protocol parsing on
  non-macOS hosts

### Phase 4: Harness validation

- add a macOS-only `examples/v1.4` scenario or smoke target
- verify:
  - boot reaches readiness
  - `VmHandle::exec("/bin/true", ...)` works
  - PTY opens over the existing SSH bridge
  - host mount is visible in guest
  - serial log and readiness artifacts are captured

Testing expectations:

- non-macOS CI can still compile the Rust-side config/handle/control-path code
  behind lightweight mocks or platform-gated unit tests
- live VM integration tests require macOS runners with the helper already built
  and signed with the required entitlement
- `vz-runner` should also have its own Swift-side unit coverage for config
  decoding, control protocol handling, and readiness-state transitions

## Feature Parity Evaluation Against CH

The long-term requirement is stronger than "a usable macOS backend." The Vz
path must satisfy the same product contracts the current CH/Motlie path
already proves in `feature/vmm`:

- guest lifecycle parity:
  - `prepare()`
  - `boot()`
  - `VmHandle::ready(...)`
  - `VmHandle::exec(...)`
  - `VmHandle::open_pty(...)`
  - `VmHandle::shutdown()`
- outbound internet egress
- SSH ingress through the localhost proxy
- auto-provisioning from incoming SSH principals
- per-guest mount visibility and isolation
- multi-tenant guest provisioning and stable guest reuse

The current Vz design is directionally correct, but it is not yet full parity
with the CH/Motlie stack. The main gaps are below.

### 1. VNet Integration

#### What Works As-Is

- The host-side SSH proxy and programmatic control plane already live above the
  hypervisor in `libs/vmm/src/ssh.rs`.
- The guest-visible contract for SSH/control is already a guest-initiated
  vsock stream on port `2222`.
- The auto-provisioning resolver path in `libs/vmm/src/provisioning.rs` is
  backend-agnostic once a guest is booted and reachable through the control
  plane.
- The "no persistent host network configuration changes" requirement is still
  compatible with the Vz direction; the missing piece is policy-capable egress,
  not host-impact minimization.

#### What Does Not Have Parity Yet

- The current CH path proves outbound internet through `libs/vnet` and
  `MotlieVnetBacking`, not merely "some form of guest networking."
- `libs/vnet` today is Linux/CH-specific:
  - it exposes a vhost-user-net backend for Cloud Hypervisor
  - it assumes a Unix socket presented to a virtio-net frontend
  - it is documented as the owner of outbound egress in the composed harness
- Apple NAT by itself is now explicitly out of scope as the parity solution
  because it cannot satisfy the vnet policy-engine requirements from `#133`:
  - no Rust-owned DNS query/response interception
  - no Rust-owned TCP connect interception
  - no shared `motlie-vnet` policy callback surface
  - no shared domain-to-IP reverse map / deny-forging behavior
- Bridged networking parity is also missing:
  - CH currently has a TAP-based admin option
  - Vz bridged networking is deferred and requires the
    `com.apple.vm.networking` entitlement
  - `VZVmnetNetworkDeviceAttachment` is not a practical parity path because it
    is macOS 26+ only

#### Required Changes

- `libs/vmm` does need a backend-neutral network intent/observability layer so
  both backends satisfy the same product requirement through different
  realizations.
- `libs/vnet` remains the canonical reusable home for DNS/TCP observability and
  egress policy control.
- Required `libs/vmm` changes:
  - replace CH-specific "vnet socket implies egress readiness" assumptions with
    backend-specific network observability
  - separate "guest has outbound egress" from "guest is using `motlie-vnet`"
  - teach harness checks to validate backend-neutral outcomes:
    - default route exists
    - DNS works
    - HTTPS fetch works
  - make policy-capable egress, not merely "some internet path", part of the
    parity contract
- Required `libs/vnet` / egress-architecture change:
  - define how a Vz guest reaches a reusable Rust-owned egress engine with the
    same policy surface as `motlie-vnet`
  - the concrete Vz raw-packet attachment for that parity path should be
    `VZFileHandleNetworkDeviceAttachment`
  - do not bury that behavior inside Apple NAT where Rust cannot observe or
    control DNS/TCP intent decisions

#### Blocking Items

- No phase-1 design yet for bridged/admin-network parity on macOS
- No reviewed backend-neutral replacement for CH-specific `vnet` artifacts in
  harness assertions and observability

### 2. VFS Integration

#### What Works As-Is

- `GuestMountSpec` already carries the key guest-facing shape:
  - mount tag
  - guest path
  - host path
- The CH orchestrator already treats filesystem backing as an injected runtime
  concern through `FilesystemBacking`.
- The current product-level requirement is "the guest sees the right mount
  points and isolation boundaries," not "the backend must literally use the
  same transport."
- Vz VirtioFS can expose host directories directly with matching tags, so
  static host-directory sharing can satisfy the same guest-visible mount
  contract.

#### What Does Not Have Parity Yet

- The current CH/Motlie path uses `libs/vfs` as a managed filesystem service:
  - host-side `FsServer`
  - vsock transport handshake by mount tag
  - policy hook (`PolicyFn`)
  - overlay-capable server behavior
- The current Vz design only covers direct VirtioFS sharing. That is not full
  parity with CH because it bypasses:
  - the host-managed `FsServer`
  - the existing policy engine in `libs/vfs/src/core/policy.rs`
  - overlay-managed mutation/event semantics
  - the current readiness rule that waits for all required mount tags to attach
- If the product requires the same managed filesystem semantics as CH, pure
  VirtioFS pass-through is insufficient.

#### Required Changes

- VFS parity is defined as the full managed-filesystem semantics documented in
  `libs/vfs/docs/DESIGN_XBACKENDS.md`, not as simple host-directory visibility.
- The parity-preserving Vz design is therefore:
  - keep the current `libs/vfs` host service in `FilesystemBacking::MotlieVfs`
  - keep the guest mounter binary and `mounts.yaml`
  - keep the host `FsServer` in the request path for guest I/O
  - use a Vz-specific guestfs transport instead of replacing guestfs with raw
    VirtioFS
- `VirtioFS` may still be useful for narrow bootstrap/debug sharing, but it is
  not the parity path.
- That implies `libs/vfs` itself should not need a policy-engine rewrite, but
  it does need transport abstraction if the existing implementation is too
  tightly coupled to the current CH-style Unix-socket-vsock path.

#### Likely `libs/vfs` Work

- factor transport assumptions so the guestfs server/client can run over:
  - current CH-style host UDS bridge
  - Vz host-side virtio-socket listeners
- preserve mount-tag handshake and `PolicyFn` behavior regardless of hypervisor

#### Blocking Items

- Product parity is defined.
- The remaining VFS uncertainty is feasibility:
  - whether Apple Vz exposes a clean enough host/guest stream path to carry the
    existing managed guestfs semantics end to end
  - whether that path preserves the same readiness and lifecycle guarantees

### 3. Guest Bootup Lifecycle

#### What Works As-Is

- `prepare()` and `boot()` already separate pure guest shaping from backend
  realization.
- `VmHandle` is already the common lifecycle surface for both backends.
- `exec`, `open_pty`, and auto-provisioning all run through the backend-neutral
  SSH control plane once the guest bridge is connected.
- The helper-process model matches the CH shell backend pattern well enough to
  preserve `pid`, shutdown fallback, and filesystem-log observability.

#### What Does Not Have Parity Yet

- The CH shell backend today is simpler than the proposed Vz helper because it
  shells out to an existing launcher script and relies on CH’s own host socket
  model.
- The Vz helper still has unimplemented parity-critical responsibilities:
  - host-side virtio-socket listener management
  - cloud-init disk creation and attachment
  - backend-specific readiness/status reporting
  - structured propagation of launch/configuration errors
- `VmHandle::ready(...)` parity is not complete until the backend-neutral
  readiness probe lands and the Vz path proves the same stage ordering:
  - backend ready
  - filesystem ready when applicable
  - SSH bridge ready
  - exec ready

#### Required Changes

- `libs/vmm` must add backend-neutral readiness semantics exactly as described
  earlier in this document
- `vz-runner` must implement:
  - a real status/control protocol
  - two-phase readiness
  - clean shutdown reporting
  - bridge/error propagation strong enough to classify failures the same way the
    harness does for CH

#### Blocking Items

- None architecturally, but the helper is still the critical implementation
  risk because several parity features currently collapse into it

### 4. Userspace Management, Provisioning, And Isolation

#### What Works As-Is

- `libs/vmm/src/provisioning.rs` is already mostly backend-neutral:
  - principal -> `GuestSpec` creation
  - stable guest reuse
  - allocator-driven slot/CID/IP assignment
  - host seeding hooks
  - `ensure_guest_for_principal(...)`
- SSH key/cert policy is also already hypervisor-neutral at the library layer:
  - guest `GuestSshAccess`
  - CA-signed ephemeral auth
  - principal resolver integration in `ssh.rs`
- Multi-tenant isolation at the orchestrator layer already exists:
  - one provisioned record per principal
  - per-guest runtime paths
  - per-guest mount roots and network assignment

#### What Does Not Have Parity Yet

- The current Vz design only partially describes how guest image customization
  carries over:
  - user account creation
  - SSH authorized principals/key injection
  - agent-state/home/workspace mount wiring
  - boot-time services such as the guest vsock SSH loop and guestfs mounter
- CH currently proves this through the existing image and cloud-init path in
  `examples/v1.4/build-guest.sh`.
- Vz must either reuse that exact guest image flow or define a reviewed macOS
  equivalent that produces the same guest-side services.

#### Required Changes

- No major `libs/vfs` or `libs/vnet` API work is required for provisioning
  itself.
- `libs/vmm` does need to state explicitly that the Vz path reuses the same
  guest image contents and cloud-init provisioning semantics as CH:
  - same users
  - same CA trust model
  - same `motlie-vmm-vsock-ssh` guest service
  - same guestfs/agent-state/bootstrap units where required

#### Blocking Items

- Until the Vz image build path is defined as "same guest image contract,
  different hypervisor realization," userspace-management parity remains
  under-specified
- `DESIGN_GUEST_IMAGE.md` is the owning prerequisite for that work

### 5. SSH Proxy And Auto-Provision Parity

#### What Works As-Is

- The hardened SSH proxy path in `libs/vmm/src/ssh.rs` is already designed to
  sit above the hypervisor:
  - localhost russh ingress
  - principal resolver hook
  - guest bridge registry
  - exec and PTY over the same authenticated guest bridge
- `GuestProvisioner::ssh_principal_resolver()` already resolves or provisions a
  guest before the proxy opens the guest channel.
- That means the auto-provision contract can remain identical for Vz if the Vz
  backend delivers the same bridge semantics and readiness ordering.

#### What Does Not Have Parity Yet

- The current Vz design mentions the bridge, but not the stronger guarantee now
  required by the hardened auto-provision path:
  - first-contact SSH must block until the guest is provisioned, booted, and
    the bridge is authenticated
  - repeat SSH to the same principal must reuse the same live guest record
  - turning auto-provision off must fail unknown principals without accidentally
    creating Vz guests
- Those rules are enforced in `ssh.rs` and provisioning today, but Vz still has
  to preserve the same observable timing and failure modes.

#### Required Changes

- No new `libs/vnet` changes are required for SSH proxy parity
- No new `libs/vfs` changes are required for SSH proxy parity directly
- `vz-runner` must expose enough status to let Rust distinguish:
  - VM running but guest bridge not ready
  - guest bridge ready
  - guest bridge failed
- `libs/vmm` harness/docs should add an explicit Vz parity requirement:
  - the existing `auto-provision-ssh` scenario must pass unchanged against a Vz
    backend selection, not just against CH

#### Acceptance Criteria

- `examples/v1.4/scenarios/auto-provision-ssh.json` must pass unchanged on Vz
- `examples/v1.4/integration/repl-auto-provision-smoke.sh` must pass unchanged
  on Vz

## Parity Summary

### Works As-Is

- backend-neutral provisioning core in `libs/vmm`
- backend-neutral SSH proxy / exec / PTY control surface in `libs/vmm`
- Vz can support vsock-based control-plane transport without redesigning the
  proxy architecture
- guest-visible static host-directory sharing is possible without changing
  `GuestMountSpec`

### Needs `libs/vmm` Changes

- backend-neutral readiness and network/filesystem observability
- explicit parity requirement that Vz passes the same lifecycle and
  auto-provision acceptance scenarios as CH
- explicit reuse of the same guest image / cloud-init / guest-service contract
- backend-neutral egress assertions in the harness instead of CH-specific
  `motlie-vnet` assumptions

### Likely Needs `libs/vfs` Changes

- only if Vz must preserve the current managed-filesystem semantics rather than
  downgraded static VirtioFS sharing
- likely transport abstraction so the same host `FsServer` / policy model can
  ride a Vz host-side virtio-socket path

### Likely Needs `libs/vnet` Changes

- Vz parity now likely requires `libs/vnet` or a sibling reusable egress layer
  to expose the same policy-capable service to non-CH backends
- the key requirement is preserving the `#133` policy/observability surface,
  not preserving the exact current vhost-user transport
- a transport abstraction or higher-level reusable egress facade is more likely
  than a new Vz-specific NAT module inside `libs/vnet`

### Blocked Or Still Under-Specified

- bridged/admin-network parity on macOS
- VFS parity is defined as full managed-filesystem semantics in
  `libs/vfs/docs/DESIGN_XBACKENDS.md`; what remains unproven is whether the
  `v1.15` guestfs PoC can actually preserve that managed path on Vz

If `v1.15` fails to preserve managed guestfs semantics:

- Vz falls back only to explicitly degraded static `VirtioFS` sharing for
  bootstrap/debug use
- that fallback does not count as `motlie-vfs` parity
- full `v1.45` parity remains blocked until a different managed transport path
  is designed and proven

## Explicit Non-Goals for the First Slice

- macOS guests
- GUI display integration
- Rosetta guest acceleration
- bridged networking
- vmnet custom networks
- live reconfiguration / hotplug
- guest metrics beyond basic process/log observability
- replacing the Rust SSH/control-plane path

## Follow-Up API Work

The Vz design exposes two real portability gaps that should be handled after
the first backend lands:

1. `GuestStorage` is too CH-shaped.
   - It needs a portable disk-attachment model or a backend-specific storage
     sub-structure.

2. `NetworkModes` is too CH/Motlie-shaped.
   - It should evolve into backend-independent intent, with backend-specific
     realization and validation.

These are follow-ups, not blockers, because a useful Vz slice can still ship
with a narrower supported subset.

## References

Primary external references consulted for this design on 2026-04-12:

- Apple Developer Documentation:
  - `VZLinuxBootLoader`
    - <https://developer.apple.com/documentation/virtualization/vzlinuxbootloader>
  - `VZNATNetworkDeviceAttachment`
    - <https://developer.apple.com/documentation/virtualization/vznatnetworkdeviceattachment/init()>
  - `VZFileHandleSerialPortAttachment`
    - <https://developer.apple.com/documentation/virtualization/vzfilehandleserialportattachment/init(filehandleforreading:filehandleforwriting:)>
  - `Sockets`
    - <https://developer.apple.com/documentation/virtualization/sockets>
  - `VZVirtualMachineConfiguration.platform`
    - <https://developer.apple.com/documentation/virtualization/vzvirtualmachineconfiguration/platform>
  - `VZMultipleDirectoryShare`
    - <https://developer.apple.com/documentation/virtualization/vzmultipledirectoryshare/init(directories:)>
  - `VZVirtioFileSystemDeviceConfiguration`
    - <https://developer.apple.com/documentation/virtualization/vzvirtiofilesystemdeviceconfiguration>
  - `VZSharedDirectory`
    - <https://developer.apple.com/documentation/virtualization/vzshareddirectory>
  - `VZBridgedNetworkDeviceAttachment`
    - <https://developer.apple.com/documentation/virtualization/vzbridgednetworkdeviceattachment>
  - `VZVmnetNetworkDeviceAttachment`
    - <https://developer.apple.com/documentation/virtualization/vzvmnetnetworkdeviceattachment>
- Rust docs:
  - `virtualization-rs 0.1.2`
    - <https://docs.rs/virtualization-rs/latest/virtualization_rs/>
  - `objc2-virtualization 0.3.2`
    - <https://docs.rs/objc2-virtualization/latest/objc2_virtualization/>
  - `objc2-virtualization` auto-trait example showing `!Send` / `!Sync`
    - <https://docs.rs/objc2-virtualization/latest/objc2_virtualization/struct.VZNetworkDevice.html>
