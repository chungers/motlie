# motlie-vmm: Virtualization.framework Backend Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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
  - `observability()` treats `api_socket` as a required run artifact
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
  - `VZLinuxBootLoader`
  - `VZVirtioFileSystemDeviceConfiguration`
  - `VZVirtioSocketDeviceConfiguration`
  - `VZNATNetworkDeviceAttachment`
  - `VZFileHandleSerialPortAttachment`
  - `VZVirtualMachineConfiguration.platform = VZGenericPlatformConfiguration`
- Apple also exposes higher-end options we can defer:
  - `VZBridgedNetworkDeviceAttachment`
  - `VZVmnetNetworkDeviceAttachment`
  - graphics, ballooning, Rosetta directory sharing, NBD, USB
- The older `virtualization-rs` crate is not a good primary choice:
  - docs.rs shows version `0.1.2`
  - docs.rs lists only `x86_64-apple-darwin`
  - it uses the older `objc` crate bindings
- The current Rust binding surface is `objc2-virtualization`:
  - docs.rs shows version `0.3.2`
  - it exposes the needed Vz classes and protocols
  - docs.rs also shows Virtualization objects such as `VZNetworkDevice` as
    `!Send` / `!Sync`

The `!Send` / `!Sync` point matters because the current orchestrator stores
backend state inside `VmHandle`, and those handles are expected to be usable
from ordinary async Rust code.

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
  - if present, backend should reject it with a clear unsupported error until a
    reviewed EFI story is added

### CPU / Memory

Current `GuestResources` maps directly:

- `boot_vcpus` -> `VZVirtualMachineConfiguration.cpuCount`
- `memory_mib` -> `memorySize = memory_mib * 1024 * 1024`
- `max_vcpus`
  - no phase-1 use
  - ignore or reject until hotplug / topology policy exists

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

Inside the guest, the existing cloud-init `mounts.yaml` flow can stay in place.
The backend only needs to guarantee that the host shares exist with matching
tags before boot.

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

## `vz-runner` Process Model

The helper should take a fully materialized launch description from Rust and
own the VM until exit.

### Rust Side

`VzBackend::boot(prepared)` should:

1. create launch/runtime directories
2. materialize a small JSON config file in `runtime_paths.launch_dir`
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
6. mark readiness by creating `vz-ready`
7. service a tiny control socket until the VM exits or shutdown is requested

### Suggested On-Disk Artifacts

Under `runtime_paths.launch_dir`:

- `vz-config.json`
- `vz-ready`
- `vz-control.sock`
- `launch.log`
- `serial.log`

Under `runtime_paths.runtime_dir`:

- backend-private transient state if needed

This is intentionally parallel to the CH shell backend’s file-oriented
observability.

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
  - unchanged

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

### Phase 3: Rust integration

- implement `backend::vz::VzBackend`
- wire `HypervisorBacking::AppleVirtualization` to the backend
- add tests for handle kind, shutdown behavior, and readiness probe behavior

### Phase 4: Harness validation

- add a macOS-only `examples/v1.4` scenario or smoke target
- verify:
  - boot reaches readiness
  - `VmHandle::exec("/bin/true", ...)` works
  - PTY opens over the existing SSH bridge
  - host mount is visible in guest
  - serial log and readiness artifacts are captured

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
- Rust docs:
  - `virtualization-rs 0.1.2`
    - <https://docs.rs/virtualization-rs/latest/virtualization_rs/>
  - `objc2-virtualization 0.3.2`
    - <https://docs.rs/objc2-virtualization/latest/objc2_virtualization/>
  - `objc2-virtualization` auto-trait example showing `!Send` / `!Sync`
    - <https://docs.rs/objc2-virtualization/latest/objc2_virtualization/struct.VZNetworkDevice.html>
