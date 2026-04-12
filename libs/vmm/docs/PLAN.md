# motlie-vmm Harness Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-08 | @codex | Address PR 140 review items: remove the dead `VmBackend` / `BackendSet` transitional layer, tighten shutdown/readiness/terminal correctness, and update the plan language to match the direct enum-dispatch runtime that is now in code |
| 2026-04-08 | @codex | Add a switchable harness terminal backend, make `shadow-terminal` the default PTY/TUI renderer with `vt100` as an explicit fallback, and record that GIF/PNG/movie output stays deferred outside `v1.4` |
| 2026-04-08 | @codex | Add asciicast export to the PTY artifact plan and standardize the scope boundary: NDJSON transcript + VTE screen JSON remain canonical, asciicast is the portable replay export, and PNG/GIF/movie generation is explicitly deferred out of `v1.4` scope |
| 2026-04-08 | @codex | Replace the 7-slot allocator with computed subnet-pool capacity, expose allocator config through `harness_v1_4`, add a file-backed scenario driver with action/expectation steps, and complete PTY/VTE capture with raw transcript NDJSON plus rendered screen JSON |
| 2026-04-07 | @codex | Complete the remaining Phase 4/5 observability slice: `VmObservability` now carries typed run-bundle metadata and capture paths, `harness_v1_4` persists PTY transcripts and internal result artifacts under the bundle root, result JSON now has structured status/error classification for agents/CI, and the PTY scenario emits hardened structured evidence |
| 2026-04-07 | @codex | Implement the first concrete Phase 4/5 slice: `observability.rs`, `VmHandle::observability()`, and `harness_v1_4 --result-json ...` for machine-readable `smoke` results; PTY result hardening remains open |
| 2026-04-07 | @codex | Rebase the near-term plan around the harness: defer CH internal-thread / alternate hypervisor backend work to a later `v2` track; make observability library-owned in Phase 4; add machine-readable results plus PTY/VTE recording work to Phase 5; and tighten Phase 8 around a real scenario/agent driver |
| 2026-04-07 | @codex | Start the PTY/session harness slice: `VmHandle::open_pty(...)`, `GuestPtySession`, and a first `harness_v1_4 pty` scenario now compile; next step is to absorb the old REPL into the harness as interactive/manual mode over the same API |
| 2026-04-07 | @codex | Implement reviewed `Runtime` injection in code: `orchestrator.rs` now composes hypervisor/filesystem/network/control-plane backing through one injected runtime, and Motlie guest backing moved behind `backend::motlie::*` adapters |
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
- an interactive/manual harness mode suitable for humans and coding agents
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
4. the primary ad-hoc/manual operator surface, replacing the standalone `v1.4`
   REPL over time

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
  - [x] `hypervisor: HypervisorBacking`
  - [x] `filesystem: FilesystemBacking`
  - [x] `network: NetworkBacking`
  - [x] `control_plane: ControlPlaneBacking`
- [ ] `examples/v1.4/harness` becomes the primary driver:
  - [x] `smoke` scenario
  - [x] `pty` scenario
  - [ ] multi-guest named scenarios
  - [ ] interactive/manual shell mode
  - [x] transcript/log bundle capture
  - [ ] action/expectation script format

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
  - [x] `BackendHandle`
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
- [ ] record the CH adapter boundary for later `v2` work:
  - [ ] reviewed `to_ch_vm_config(...)` boundary
  - [ ] preserve a future path from shell launch to in-process
        `VmConfig` + `start_vmm_thread(...)`
- [x] make backend dispatch enum-based, not dynamically discovered
- [x] inject backend dispatch into generic orchestrator code
- [x] converge the earlier transitional backend injection to reviewed
      `Runtime` injection, then remove the dead transitional scaffolding
- [x] add the first backend implementation:
  - [x] `ChShellBackend`
  - [x] keep it close enough to current `v1.3` shell/CLI behavior to boot and
        validate the `v1.4` harness
- [x] apply the same injection rule to guest backing providers:
  - [x] `motlie-vfs`
  - [x] `motlie-vnet`
  - [x] SSH bridge/control plane
- [x] express the reviewed runtime composition explicitly in code:
  - [x] `Runtime`
  - [x] `HypervisorBacking`
  - [x] `FilesystemBacking`
  - [x] `NetworkBacking`
  - [x] `ControlPlaneBacking`

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
- CH internal-thread / fork-exec backend work is explicitly outside the
  critical `v1.4` path and can land later as `v2` backend expansion once the
  harness and Motlie-backed flow are complete

## Phase 4: GuestFS and SSH Bridge Lifecycle Extraction

Goal:
- remove subsystem lifecycle wiring from the example

Tasks:
- [x] add `libs/vmm/src/guestfs.rs`
- [x] move guest provisioning and mount attachment there
- [x] move guest listener spawn logic there
- [x] move SSH bridge accept/register lifecycle out of the `v1.4` harness path
- [x] unify guest launch/SSH handle/vnet/VFS state into one library-owned handle
- [ ] add library-owned observability surfaces for the running guest lifecycle:
  - [x] launch-log path / serial-log path on the typed handle
  - [x] typed accessors for effective runtime roots and sockets
  - [x] transcript/log capture surfaces that the harness can consume without
        reimplementing path discovery
  - [x] enough typed metadata to bundle a run for later debugging

Acceptance:
- `repl_host_v1_3` does not manually coordinate SSH bridge and VFS listener state
- guest lifecycle state is tracked by the library, not by ad hoc REPL maps
- harness and future drivers can discover core run artifacts through the
  library rather than reconstructing paths ad hoc
- [x] first observability slice is library-owned through
      `VmHandle::observability()`

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
  - [x] first machine-readable result output
- [x] keep it rootless/userspace-only
- [x] make it the default substrate for building later `v1.4` phases
- [x] add the first PTY/session-driven scenario under `examples/v1.4/harness/`
- [x] move the practical ad-hoc/manual `v1.4` shell workflow into
      `harness_v1_4 shell` over the same lifecycle APIs
- [x] save the standard multi-guest bring-up as
      `examples/v1.4/setup-multiguest.harness`
- [x] add expectation-driven harness smoke scripts for:
  - [x] one multi-guest harness shell run
  - [x] two concurrent harness instances
- [x] add transcript/log bundle capture so harness runs preserve enough state
      for debugging subtle PTY, VFS, and vnet regressions
- [x] add machine-readable result output for scenarios and ad-hoc operations:
  - [x] structured pass/fail result records
  - [x] stable machine-readable guest/run metadata
  - [x] structured error classification suitable for agents and CI
- [x] add a PTY/VTE capture layer:
  - [x] keep a VTE-style screen buffer for PTY sessions so the harness can
        reason about rendered terminal state, not only raw byte streams
  - [x] expose rendered screen state to scripted expectations
  - [x] preserve PTY transcripts in a reusable artifact format
  - [x] make the terminal-state backend switchable in the harness
  - [x] default the harness PTY/TUI path to the higher-fidelity `shadow`
        backend while keeping `vt100` as a fallback/debugging mode
- [ ] evaluate terminal-recording artifact generation for human verification:
  - [ ] `ttyrec`
  - [x] `asciinema`-compatible cast export
  - [ ] `t-rec`
  - [x] decide that the harness should emit asciicast as the portable replay
        export layered on top of the canonical NDJSON transcript + screen JSON
  - [x] explicitly defer rendered movie artifacts such as GIF/PNG output out of
        `v1.4` scope

Acceptance:
- later `v1.4` work can be developed against a stable non-REPL harness
- the harness can boot, exec, and shut down a guest without depending on prompt
  parsing or human-oriented output
- the harness remains the proving ground for the full Motlie-backed path
- multi-guest scripted validation runs through the harness rather than the old
  standalone `repl_host_v1_4`
- [x] the harness can emit first-pass structured results for agents and CI
- PTY-driven scenarios can assert on rendered terminal state, not just command
  exit codes
- failed runs can preserve enough terminal/log artifacts for human
  verification

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
- [x] add `libs/vmm/src/provisioning.rs`
- [x] add a library-owned guest registry keyed by principal/guest name
- [x] define the stable allocation policy using
  `libs/vmm/src/network_alloc.rs`
- [x] make allocation include:
  - [x] slot
  - [x] vsock CID
  - [x] admin ingress subnet/IP pair
  - [x] admin MAC
  - [x] egress MAC
  - [x] vhost-user socket path
  - [x] runtime namespace roots
- [x] add typed exhaustion errors instead of silent collision-prone saturation
- [x] add orchestrator entrypoint:
  - [x] `ensure_guest_for_principal()`
- [x] make the SSH proxy call the orchestrator to resolve-or-create guests
- [x] ensure shutdown/reboot reuses existing assignments within one harness run

Acceptance:
- `ssh alice@localhost` and `ssh bob@localhost` resolve predictably to their
  guests
- `ssh jane@localhost` and `ssh mike@localhost` can create new guests on first
  contact
- assignments are stable across relaunch within the same harness run
- capacity exhaustion fails with a typed error instead of producing collisions

## Phase 8: Validation and Agent Harness Mode

Goal:
- broaden the harness from smoke driver to scenario/agent driver

Tasks:
- [ ] add `libs/vmm/src/validation.rs`
- [ ] turn current smoke tests into typed validation profiles
- [ ] add structured pass/fail output
- [x] decide that the stable harness is the `examples/v1.4/harness` binary,
      not a mode bolted onto the old REPL
- [ ] ensure it supports:
  - [x] boot-and-wait through `boot` + `handle.ready(...)`
  - [x] exec
  - [ ] validate
  - [x] shutdown-and-wait
  - [x] PTY/send/expect script steps
- [x] design and implement the stable scenario/script format:
  - [x] action/expectation pairs
  - [x] PTY send/read/resize/expect steps
  - [x] multi-guest coordination
  - [x] stable machine-readable outputs per step
- [ ] converge ad-hoc/manual operation onto the same engine:
  - [ ] harness shell commands should be thin wrappers over the scenario/driver
        engine
  - [ ] user reports should be reproducible as saved scripts, not one-off REPL
        sessions

Acceptance:
- an agent can drive the harness without depending on the REPL prompt
- validation returns machine-usable results rather than only stderr text
- the same engine supports:
  - [x] scripted regression scenarios
  - [ ] human interactive/manual operation
  - [x] ad-hoc coding-agent experimentation

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
- removing ad-hoc/manual operation entirely
- moving `motlie-vfs` or `motlie-vnet` implementation details into `libs/vmm`

## Checkpoint Rule

Extraction should remain incremental:

- keep `examples/v1.3` runnable after every phase
- prefer moving pure and typed logic first
- only move lifecycle code once the behavior is proven in the current harness
