# motlie-vmm: Virtualization.framework Backend Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-25 | @codex-vz | Link Vz backend readiness to the shared guest convergence contract: `control-plane-ready` is an interactive-readiness gate, while full VFS/VNET/egress certification remains an explicit harness validation step |
| 2026-04-17 | @vmm-vz-cdx | Record `libs/vfs/examples/v1.05` as the Tart-backed guest-image / guest-contract probe, document Tart as an interim signed launcher ahead of `vz-runner`, and make the `v1.05` -> `v1.15` sequencing explicit |
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

The cross-backend boot/provisioning contract is maintained in
[`CONVERGENCE.md`](./CONVERGENCE.md). Any Vz-specific readiness behavior must
name which contract phase it implements and must not hide full validation work
behind first-contact SSH auto-provisioning.

The design target is not feature parity with the Motlie-backed Cloud
Hypervisor stack on day one. The first Vz slice should support the same
orchestrator contracts with a narrower backend capability set:

- Linux guest boot on macOS
- SSH control plane over virtio-vsock
- host-visible serial log capture
- host directory sharing for guest mounts
- simple host-managed networking
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

## Interim `v1.05` Probe: Tart Before `vz-runner`

The current `v1.05` Apple Vz work under `libs/vfs/examples/v1.05/` is not the
final backend implementation. It is a feasibility and guest-contract probe that
uses Tart as a temporary signed launcher while Motlie's own `vz-runner`
packaging remains pending.

`v1.05` owns:

- proof that Apple Vz can boot a Linux guest on the target Apple Silicon host
- proof that a Motlie-controlled guest variant can be provisioned on top of a
  known-good Tart Ubuntu image
- proof that `motlie-vfs-guest`, `mounts.yaml`, and the guest-side systemd unit
  can be installed and validated in the guest without modifying the CH path

`v1.05` does not own:

- host-side `motlie-vfs` server integration
- a Vz guestfs transport
- managed overlay semantics
- policy-engine semantics

Those belong to the later `v1.15` slice.

The temporary Tart choice is deliberate:

- Tart is already distributed as a signed macOS app bundle that can launch
  Apple `Virtualization.framework` guests
- that lets the repo prove Linux guest viability on macOS without first
  solving the local developer signing story for Motlie's own `vz-runner`
- it mirrors the spirit of shelling out to Cloud Hypervisor during the CH
  examples, while staying Vz-only and leaving the CH path untouched

This does not change the phase-1 backend target:

- the long-term backend still uses a Motlie-owned `vz-runner`
- Tart is an interim launcher for the `v1.05` probe only

## `v1.05` / `v1.15` Sequencing

The Apple Vz decimal-versioning scheme currently means:

- `v1.05`
  - guest image / boot / guest-contract probe
  - Tart-backed
  - closest practical parallel to `libs/vfs/examples/v1` without attempting the
    transport slice yet
- `v1.15`
  - first managed guestfs transport slice on Vz
  - proves the host `motlie-vfs` server loop, guestfs connectivity, and the
    first end-to-end mounted filesystem semantics

The `v1.05` slice therefore reduces uncertainty for `v1.15`, but it is not a
substitute for `v1.15`.

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

Recommended Vz mapping:

- `AdminNetMode::None + EgressNetMode::None`
  - boot without a network device
- `AdminNetMode::None + EgressNetMode::VhostUser`
  - map to one NAT-backed virtio NIC with `VZNATNetworkDeviceAttachment`
- all other combinations
  - reject as unsupported for Vz phase 1

Reasoning:

- Vz does not consume the existing userspace `motlie-vnet` vhost-user socket
- simple NAT is the closest equivalent to “guest has outbound network access”
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
    "root_disk": "/abs/path/root.qcow2",
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
    "mode": "nat"
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
