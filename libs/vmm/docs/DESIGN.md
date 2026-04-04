# motlie-vmm: Reusable VM Orchestration Extracted from Proven Examples

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-03 | @codex | Initial DESIGN for `libs/vmm`: capture the post-`v1.2` extraction target for reusable VM orchestration code |

## Status

This document describes the intended extraction target for a future `libs/vmm`
library. It is intentionally design-only at this stage.

The near-term proving ground is `libs/vfs/examples/v1.2`, where the guest
launch flow, dual-network composition, and cloud-init/runtime assembly should
first be validated end to end before the reusable pieces are moved here.

## Problem Statement

The current example lineage (`v1`, `v1.1`, and planned `v1.2`) contains
growing amounts of host-side VM orchestration logic:

- Cloud Hypervisor argument construction
- guest-specific socket, CID, IP, and runtime-path allocation
- cloud-init asset generation
- runtime overlay assembly
- coordinated startup/shutdown of helper services such as `motlie-vfs` and
  `motlie-vnet`
- launch-mode composition such as admin ingress vs egress networking

That logic is useful beyond any single example, but extracting it too early
creates two risks:

1. we freeze the wrong API before `v1.2` proves the workflow
2. we blur crate boundaries by pushing VM orchestration into `motlie-vfs` or
   `motlie-vnet`

We need a dedicated home for reusable VM orchestration once the `v1.2` flow is
validated, while keeping the early experimentation in examples where it is easy
to change.

## Goals

- Provide a dedicated library for host-side VM orchestration and launch
  composition.
- Extract reusable code only after it is proven in `examples/v1.2`.
- Keep device-specific logic in the device crates:
  - `motlie-vfs` owns filesystem serving and guest mount transport
  - `motlie-vnet` owns outbound guest networking and the vhost-user-net backend
- Centralize Cloud Hypervisor launch construction and per-guest runtime layout.
- Centralize guest boot asset generation such as cloud-init and generated
  `mounts.yaml`.
- Make multi-guest orchestration and per-guest lifecycle management reusable.
- Let future examples become thin wiring layers over library code rather than
  carrying large shell-script control planes.

## Non-Goals

- Defining a general-purpose hypervisor abstraction across VMMs.
- Replacing `cloud-hypervisor` with a pluggable backend in v1.
- Moving `motlie-vfs` protocol logic into `libs/vmm`.
- Moving `motlie-vnet` backend logic into `libs/vmm`.
- Freezing the final public Rust API before `v1.2` validates the operator flow.
- Eliminating example scripts immediately; scripts may remain as thin wrappers.

## Design Principles

### 1. Prove First, Extract Second

`examples/v1.2` is the place to prove:

- dual-network launch composition
- guest runtime overlay layout
- cloud-init rendering rules
- `motlie-vfs` + `motlie-vnet` + Cloud Hypervisor composed startup
- operational shutdown and cleanup behavior

Only code that is stable after that validation should move into `libs/vmm`.

### 2. Keep Ownership Boundaries Sharp

`libs/vmm` should own orchestration, not device behavior.

- `motlie-vfs` remains the filesystem subsystem
- `motlie-vnet` remains the networking subsystem
- `libs/vmm` composes subsystems into a guest launch/runtime

This avoids turning `motlie-vfs` or `motlie-vnet` into grab-bag crates for
unrelated orchestration concerns.

### 3. Prefer Typed Host-Side Models Over Shell State

If `v1.2` proves that a piece of shell glue is fundamental to the runtime
contract, it should move into typed Rust structures:

- guest identity
- runtime directories
- socket allocation
- NIC configuration
- cloud-init fragments
- launch-time artifact selection

The example can still expose an operator-friendly CLI, but the hard parts
should not remain encoded as ad hoc environment variables and string assembly.

## Proposed Responsibility Split

### `motlie-vfs`

Owns:

- `FsServer`
- guest mount protocol
- guest mounter binaries
- host REPL commands that are intrinsically about mounted filesystems

Does not own:

- Cloud Hypervisor command construction
- network topology selection
- cloud-init rendering as a general facility

### `motlie-vnet`

Owns:

- libslirp wrapper
- vhost-user-net backend
- outbound DHCP/DNS/internet egress mechanics
- optional hostfwd helper behavior

Does not own:

- guest persona or cloud-init rendering
- VM overlay layout
- VMM lifecycle orchestration

### `libs/vmm`

Owns:

- guest launch configuration
- per-guest runtime directory layout
- generated cloud-init / boot assets
- CH command-line/device composition
- orchestration of optional subsystems (`motlie-vfs`, `motlie-vnet`)
- deterministic startup/shutdown and cleanup coordination

## General Layout

The exact module split should follow what `v1.2` proves, but the expected shape
is:

```text
libs/vmm/
  docs/
    DESIGN.md
  src/
    lib.rs
    guest.rs        # typed guest identity/config model
    runtime.rs      # runtime dir layout, sockets, overlays, temp assets
    cloud_init.rs   # render user-data/meta-data/network-config
    network.rs      # admin-net / egress-net composition model
    launcher.rs     # CH argument construction and process launch
    orchestrator.rs # high-level "prepare, start, stop" flow
```

This is a target shape, not a commitment to specific filenames.

## Candidate Reusable Pieces to Extract from `v1.2`

### 1. Launch Composition

Expected extraction:

- admin ingress vs egress network selection
- per-mode CH args
- shared-memory requirements for vhost-user devices
- NIC MAC assignment and guest-visible route ownership

This is orchestration logic and belongs in `libs/vmm`, not in `motlie-vnet`.

### 2. Guest Boot Asset Generation

Expected extraction:

- cloud-init `user-data`
- cloud-init `meta-data`
- optional `network-config`
- generated `mounts.yaml`

The critical rule is that `libs/vmm` should generate these assets from typed
guest/runtime state rather than from example-specific string templates once the
behavior is proven.

### 3. Runtime Layout and Artifact Assembly

Expected extraction:

- runtime directory naming
- overlay image paths
- cloud-init seed paths
- per-guest sockets and API sockets
- launch-helper log/serial locations

This logic is already shaping into a reusable contract in the example lineage.

### 4. Subsystem Lifecycle Wiring

Expected extraction:

- start/stop `motlie-vnet`
- coordinate `motlie-vfs` host-side provisioning with guest launch
- ensure cleanup order is deterministic on shutdown

This code should become reusable once the composed `v1.2` flow is stable.

## Initial API Direction

The eventual API should be small and orchestration-centered. A likely shape is:

```rust
pub struct GuestSpec { ... }
pub struct RuntimeSpec { ... }
pub struct LaunchArtifacts { ... }

pub struct VmOrchestrator { ... }

impl VmOrchestrator {
    pub fn prepare(&self, guest: &GuestSpec) -> Result<LaunchArtifacts>;
    pub fn launch(&self, guest: &GuestSpec, artifacts: &LaunchArtifacts) -> Result<VmHandle>;
}
```

This is directionally useful, but should not be implemented until `v1.2`
answers the concrete lifecycle questions.

## Relationship to `examples/v1.2`

`examples/v1.2` should initially own the fluid implementation details.

Once validated, the extraction sequence should be:

1. move pure rendering/layout logic into `libs/vmm`
2. move CH launch construction into `libs/vmm`
3. leave example-specific topology and sample operator UX in `examples/v1.2`
4. make `examples/v1.2` consume `libs/vmm` instead of owning the orchestration

This keeps the example as a demo/runbook while shrinking its bespoke logic.

## Validation Gates Before Extraction

Before code moves from `v1.2` into `libs/vmm`, the `v1.2` flow should prove:

- guest boots reliably with generated launch assets
- admin ingress and egress networking compose correctly
- `motlie-vfs` mounts still work in the composed launch
- guest has outbound internet access through `motlie-vnet`
- startup/shutdown sequencing is deterministic
- Alice/Bob multi-guest flows do not rely on accidental path or ordering hacks

If any of those remain unstable, the code should stay in the example until the
behavior settles.

## Open Questions

- Should `libs/vmm` be Cloud Hypervisor-specific in v1, or merely CH-first with
  internal seams for future expansion?
- How much of the current REPL launch flow should remain example-only versus
  move into reusable orchestration helpers?
- Should cloud-init generation live as a standalone submodule, or remain an
  implementation detail of launch preparation?
- How should guest runtime artifact cleanup be exposed: explicit handle API,
  best-effort `Drop`, or both?

## Immediate Next Step

Do not implement `libs/vmm` yet.

Use `examples/v1.2` to prove the runtime behavior first, then extract the
stable orchestration and rendering pieces into `libs/vmm` with this document as
the boundary guide.
