# motlie-vmm Harness Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex | Lock the reviewed `Runtime { hypervisor, filesystem, network, control_plane }` model: `BackendSet` remains only as an intermediate implementation step, and the next cleanup is to inject Motlie guest backing through the same boundary |
| 2026-04-07 | @codex | Inject VM backend dispatch through `BackendSet` so generic orchestrator code no longer imports concrete CH backend modules directly; record Motlie guest backing injection as the next cleanup step |
| 2026-04-07 | @codex | Start refactoring `v1.4` toward the reviewed backend hierarchy: `backend/mod.rs`, `backend/ch/`, placeholder `backend/motlie/`, placeholder `backend/vz/`, and explicitly document the simple CH guest path as a desired outcome |
| 2026-04-07 | @codex | Complete the first usable `v1.4` lifecycle API: library-owned guestfs and SSH bridge services, `VmHandle::exec(...)`, rootless harness validation, and child-handle-based shutdown/readiness |
| 2026-04-07 | @codex | Finish Phase 1/2 API convergence in code and start Phase 3 with `backend.rs`, `orchestrator.rs`, `PrepareRequest`, `prepare()`, `boot()`, `VmHandle`, and the initial `ChShellBackend` |
| 2026-04-07 | @codex | Record the CH v44.0 internal API alignment plan: keep `GuestUser` and `GuestSshAccess` above the adapter layer, and split CH-shaped inputs into `GuestResources`, `GuestStorage`, and `BootArtifacts` |
| 2026-04-07 | @codex | Align the reviewed `v1.4` API around `GuestUser`, `GuestSshAccess`, explicit CA-issued credentials, and `boot()` plus handle-based readiness |
| 2026-04-07 | @codex | Start Phase 2 extraction in `libs/vmm/src/artifacts.rs` and make it the explicit owner of rendered boot/runtime artifacts |
| 2026-04-07 | @codex | Add an explicit embedded-image / union-binary prototype phase after harness bootstrap |
| 2026-04-07 | @codex | Insert an explicit programmatic harness bootstrap phase after lifecycle extraction so later `v1.4` phases can build on a stable non-REPL substrate |
| 2026-04-07 | @codex | Add an explicit auto-provisioning phase for new SSH principals and document the library-owned guest allocation policy |
| 2026-04-07 | @codex | Add an explicit reporting/metrics phase for `v1.4`, using Cloud Hypervisor host-side API/event data plus guest-side SSH probes |
| 2026-04-06 | @codex | Add explicit post-checkpoint plan for extracting a stable automation harness from `examples/v1.3` into reusable `libs/vmm/src` modules |

## Status

Current source of truth for the runnable proving ground:

- implementation: `libs/vmm/examples/v1.3/`
- design: `libs/vmm/docs/DESIGN.md`
- checkpoint PR: `dev/vmm-v1.3` -> `feature/vmm`

What is already working in the current `v1.3` checkpoint:

- Cloud Hypervisor guest launch through the VMM-owned example harness
- VFS guest mount composition for home, workspace, and agent-state
- motlie-vnet egress backend integration
- SSH CA injection and CA-based guest login
- SSH proxy ingress on `localhost:2222`
- programmatic exec over the same SSH proxy
- PTY correctness fixes for interactive and exec paths
- deterministic shutdown fallback (API -> SIGTERM -> SIGKILL)

What is still missing for a polished, reusable harness:

- blocking launch with explicit readiness gates
- library-owned lifecycle state instead of example-owned maps
- typed validation/reporting APIs
- a non-interactive harness mode suitable for agents and CI
- a stable host-side + guest-side reporting surface for CPU/memory/disk/network
  visibility
- automatic guest provisioning when a new SSH principal appears
- an explicit programmatic harness layer that later phases can target directly
- a practical single-binary distribution prototype for the curated guest image

## Objective

Turn the working `v1.3` example into:

1. a thin interactive REPL for humans
2. a reusable library for guest orchestration
3. a stable runnable harness for agents and future Motlie development

## Desired Outcomes

- [ ] `v1.4` Motlie-backed path works end to end:
  - [ ] a runnable harness uses Motlie guest backing providers
  - [ ] scripted validation checks guest behavior end to end
- [ ] `motlie-vmm` also supports a simple standard guest path:
  - [ ] no Motlie guestfs backing
  - [ ] no Motlie userspace vnet backing
  - [ ] a small Cloud Hypervisor “hello world” example boots a guest through
        the same lifecycle API using ordinary hypervisor-managed resources
- [ ] reviewed `Runtime` composition becomes the injected runtime contract:
  - [ ] `hypervisor: HypervisorBacking`
  - [ ] `filesystem: FilesystemBacking`
  - [ ] `network: NetworkBacking`
  - [ ] `control_plane: ControlPlaneBacking`

The rule for this plan is:

- preserve the current working `v1.3` example behavior
- extract stable lifecycle logic into `libs/vmm/src`
- keep operator UX and demo-specific wiring in the example

## Phase 1: Typed Spec and Network Extraction

Goal:
- remove pure configuration and allocation policy from `repl_host_v1_3`

Tasks:
- [x] add `libs/vmm/src/spec.rs`
- [x] move guest/mount/user/SSH-access config types out of `repl_host.rs`
- [x] add `libs/vmm/src/network.rs`
- [x] move `AdminNet`, `EgressNet`, and guest CID/IP/MAC allocation there
- [x] expose typed helpers for socket paths and network mode validation
- [x] replace provisional review naming with:
  - [x] `GuestUser`
  - [x] `GuestSshAccess`
  - [x] `GuestResources`
  - [x] `GuestStorage`
  - [x] `BootArtifacts`
  - [x] `SoftwareProfile`
- [x] make the reviewed `GuestSpec` shape explicit:
  - [x] `guest_id`
  - [x] `hostname`
  - [x] `user`
  - [x] `ssh`
  - [x] `storage`
  - [x] `boot`
- [x] document the CA binding surface:
  - [x] `SshCa::issue_guest_ssh_credentials(&GuestUser, &GuestSshAccess)`

Acceptance:
- `repl_host_v1_3` compiles using library-owned spec/network types
- network allocation remains stable across shutdown/relaunch in one harness run

## Phase 2: Launch Artifact and Runtime Layout Extraction

Goal:
- move render/layout code into reusable library modules

Owning module:
- `libs/vmm/src/artifacts.rs`

Tasks:
- [x] add `libs/vmm/src/artifacts.rs`
- [x] move cloud-init rendering there
- [x] move mounts.yaml rendering there
- [x] move runtime path helpers there
- [x] move launch script rendering there
- [x] make `artifacts.rs` consume reviewed declarative inputs:
  - [x] `SoftwareProfile`
  - [x] `GuestStorage`
  - [x] `BootArtifacts`

Acceptance:
- `repl_host_v1_3` no longer owns cloud-init/mounts/layout string generation
- launch assets are testable without running the REPL
- image/build artifact handling is stabilized enough for the later union-binary
  prototype phase

## Phase 3: Orchestrator and Blocking Readiness

Goal:
- create a typed lifecycle API suitable for agents

Tasks:
- [x] add `libs/vmm/src/orchestrator.rs`
- [x] add `libs/vmm/src/backend/mod.rs`
- [x] move backend-specific implementation into `libs/vmm/src/backend/`
- [x] establish vertical backend hierarchy:
  - [x] `backend::ch`
  - [x] placeholder `backend::motlie`
  - [x] placeholder `backend::vz`
- [x] define `PreparedGuest`, `VmHandle`, `ShutdownReport`
- [x] define backend review types:
  - [x] `BackendKind`
  - [x] `VmBackendCapabilities`
  - [x] `VmBackend`
  - [x] intermediate `BackendSet`
- [x] add `prepare()`
- [x] add `boot()`
- [x] add handle-based readiness:
  - [x] `VmHandle::ready(&ReadinessPolicy)`
- [x] add handle-based exec:
  - [x] `VmHandle::exec(...)`
- [x] add handle-based shutdown:
  - [x] `VmHandle::shutdown()`
- [x] add explicit readiness gates:
  - [x] API socket ready
  - [x] guestfs connected
  - [x] SSH bridge connected
  - [x] exec-ready probe
- [x] make CH-shaped boot inputs flow through:
  - [x] `GuestResources`
  - [x] `GuestStorage`
  - [x] `BootArtifacts`
- [ ] add an explicit CH adapter plan:
  - [ ] reviewed `to_ch_vm_config(...)` boundary
  - [ ] preserve ability to switch from CLI launch to in-process
        `VmConfig` + `start_vmm_thread(...)`
- [x] make backend dispatch enum-based, not dynamically discovered
- [x] inject backend dispatch into generic orchestrator code
- [ ] converge the intermediate `BackendSet` injection to reviewed
      `Runtime` injection
- [x] add the first backend implementation:
  - [x] `ChShellBackend`
  - [x] keep it close enough to current `v1.3` shell/CLI behavior to boot and
        validate the `v1.4` harness
- [ ] apply the same injection rule to guest backing providers:
  - [ ] `motlie-vfs`
  - [ ] `motlie-vnet`
  - [ ] SSH bridge/control plane
- [ ] express the reviewed runtime composition explicitly in code:
  - [ ] `Runtime`
  - [ ] `HypervisorBacking`
  - [ ] `FilesystemBacking`
  - [ ] `NetworkBacking`
  - [ ] `ControlPlaneBacking`

Acceptance:
- a caller can block until a guest is actually usable
- boot/readiness failures identify which stage timed out or failed
- the backend seam is generic enough for future `Vz` support without forcing a
  public API rewrite
- the backend crate layout supports both:
  - CH/VZ simple hypervisor guests
  - Motlie-backed guest providers as explicit vertical slices
- the generic lifecycle API accepts an injected reviewed runtime composition,
  rather than assembling concrete CH and Motlie implementations inside
  `orchestrator.rs`

## Phase 4: GuestFS and SSH Bridge Lifecycle Extraction

Goal:
- remove subsystem lifecycle wiring from the example

Tasks:
- [x] add `libs/vmm/src/guestfs.rs`
- [x] move guest provisioning and mount attachment there
- [x] move guest listener spawn logic there
- [x] move SSH bridge accept/register lifecycle out of the `v1.4` harness path
- [x] unify guest launch/SSH handle/vnet/VFS state into one library-owned handle

Acceptance:
- `repl_host_v1_3` does not manually coordinate SSH bridge and VFS listener state
- guest lifecycle state is tracked by the library, not by ad hoc REPL maps

## Phase 5: Programmatic Harness Bootstrap

Goal:
- create a stable, non-interactive `v1.4` harness substrate that later phases
  can use directly during development and validation

Tasks:
- [x] add a `v1.4`-owned non-interactive harness entrypoint under
      `examples/v1.4/`
- [x] make it call library APIs instead of REPL command strings
- [ ] support:
  - [x] `boot`
  - [x] `handle.ready(...)`
  - [x] `exec`
  - [x] `handle.shutdown()`
  - [ ] machine-readable result output
- [x] keep it rootless/userspace-only
- [ ] make it the default substrate for building later `v1.4` phases

Acceptance:
- later `v1.4` work can be developed against a stable non-REPL harness
- the harness can boot, exec, and shut down a guest without depending on prompt
  parsing or human-oriented output
- the harness remains the proving ground for the full Motlie-backed path

## Phase 6: Embedded Image / Union Binary Prototype

Goal:
- prototype a single distributable `v1.4` binary that embeds an opinionated
  guest image and boots it from memfd-backed artifacts where practical

Tasks:
- [ ] define a special build flag for the union-binary mode
- [ ] embed curated guest image assets into the harness ELF `.rodata`
- [ ] add library support for image sources that come from:
  - [ ] normal on-disk artifacts
  - [ ] embedded bytes
- [ ] prefer memfd-backed handoff to Cloud Hypervisor using `/proc/self/fd/...`
- [ ] keep a fallback path when a boot artifact cannot be consumed directly from
      memfd
- [ ] validate the prototype through the programmatic `v1.4` harness:
  - [ ] boot
  - [ ] SSH exec
  - [ ] VFS mounts
  - [ ] outbound network
  - [ ] shutdown

Acceptance:
- the union-binary mode produces one runnable artifact
- the prototype remains userspace-only with `kvm` group membership as the main
  host requirement
- the normal development/image-on-disk mode remains available

## Phase 7: Automatic Guest Provisioning From SSH Principal

Goal:
- let the library resolve or create guests on first SSH contact without
  hardcoding a fixed guest list in the example harness

Tasks:
- [ ] add `libs/vmm/src/provisioning.rs`
- [ ] add a library-owned guest registry keyed by principal/guest name
- [ ] define the stable allocation policy using
  `libs/vmm/src/network_alloc.rs`
- [ ] make allocation include:
  - [ ] slot
  - [ ] vsock CID
  - [ ] admin ingress subnet/IP pair
  - [ ] admin MAC
  - [ ] egress MAC
  - [ ] vhost-user socket path
  - [ ] runtime namespace roots
- [ ] add typed exhaustion errors instead of silent collision-prone saturation
- [ ] add orchestrator entrypoint:
  - [ ] `ensure_guest_for_principal()`
- [ ] make the SSH proxy call the orchestrator to resolve-or-create guests
- [ ] ensure shutdown/reboot reuses existing assignments within one harness run

Acceptance:
- `ssh alice@localhost` and `ssh bob@localhost` resolve predictably to their
  guests
- `ssh jane@localhost` and `ssh mike@localhost` can create new guests on first
  contact
- assignments are stable across relaunch within the same harness run
- capacity exhaustion fails with a typed error instead of producing collisions

## Phase 8: Validation and Agent Harness Mode

Goal:
- provide a runnable, non-interactive harness for automation

Tasks:
- [ ] add `libs/vmm/src/validation.rs`
- [ ] turn current smoke tests into typed validation profiles
- [ ] add structured pass/fail output
- [ ] decide whether the stable harness is:
  - [ ] a new binary
  - [ ] or a non-interactive mode in `repl_host_v1_3`
- [ ] ensure it supports:
  - [ ] launch-and-wait
  - [ ] exec
  - [ ] validate
  - [ ] shutdown-and-wait

Acceptance:
- an agent can drive the harness without depending on the REPL prompt
- validation returns machine-usable results rather than only stderr text

## Phase 9: Polish and Hardening

Goal:
- make the harness dependable infrastructure for future Motlie work

Tasks:
- [ ] classify lifecycle errors with stable types
- [ ] add per-operation timeout controls
- [ ] add integration tests for multi-guest bring-up/teardown
- [ ] add tests for repeated relaunch of the same guest
- [ ] document the stable harness contract in `README.md`

Acceptance:
- future Motlie development can use the harness as a standard development/test substrate
- the crate also demonstrates a trivial standard hypervisor guest path through a
  small example using the same lifecycle API

## Phase 10: Guest Reporting and Metrics

Goal:
- provide a reusable, machine-readable VM reporting surface for debugging,
  automation, and future Motlie development

Tasks:
- [ ] add `libs/vmm/src/reporting.rs`
- [ ] add typed report structures for:
  - [ ] lifecycle state
  - [ ] Cloud Hypervisor host-side state
  - [ ] guest-side probe results
- [ ] add Cloud Hypervisor host-side collection through:
  - [ ] `--api-socket`
  - [ ] `--event-monitor`
  - [ ] `/api/v1/vm.info`
  - [ ] `/api/v1/vm.counters`
- [ ] add guest-side reporting over SSH exec for:
  - [ ] CPU utilization
  - [ ] guest memory usage
  - [ ] filesystem/disk usage
  - [ ] process/service health
  - [ ] outbound network reachability
- [ ] expose a reusable `report <guest>` operator flow in `v1.4`
- [ ] add a machine-readable output mode for automation

Acceptance:
- the harness can produce a structured snapshot for a running guest without
  requiring an interactive SSH session
- report output clearly distinguishes:
  - [ ] Cloud Hypervisor-visible counters/state
  - [ ] guest-OS-visible metrics
- future debugging of performance or isolation regressions can start from one
  stable reporting command rather than ad hoc shell commands

## Non-Goals for This Plan

- replacing Cloud Hypervisor with an abstract hypervisor backend
- removing the human REPL entirely
- moving `motlie-vfs` or `motlie-vnet` implementation details into `libs/vmm`

## Checkpoint Rule

Extraction should remain incremental:

- keep `examples/v1.3` runnable after every phase
- prefer moving pure and typed logic first
- only move lifecycle code once the behavior is proven in the current harness
