# motlie-vmm Harness Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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

## Objective

Turn the working `v1.3` example into:

1. a thin interactive REPL for humans
2. a reusable library for guest orchestration
3. a stable runnable harness for agents and future Motlie development

The rule for this plan is:

- preserve the current working `v1.3` example behavior
- extract stable lifecycle logic into `libs/vmm/src`
- keep operator UX and demo-specific wiring in the example

## Phase 1: Typed Spec and Network Extraction

Goal:
- remove pure configuration and allocation policy from `repl_host_v1_3`

Tasks:
- [ ] add `libs/vmm/src/spec.rs`
- [ ] move guest/mount/identity config types out of `repl_host.rs`
- [ ] add `libs/vmm/src/network.rs`
- [ ] move `AdminNet`, `EgressNet`, and guest CID/IP/MAC allocation there
- [ ] expose typed helpers for socket paths and network mode validation

Acceptance:
- `repl_host_v1_3` compiles using library-owned spec/network types
- network allocation remains stable across shutdown/relaunch in one harness run

## Phase 2: Launch Artifact and Runtime Layout Extraction

Goal:
- move render/layout code into reusable library modules

Tasks:
- [ ] add `libs/vmm/src/artifacts.rs`
- [ ] move cloud-init rendering there
- [ ] move mounts.yaml rendering there
- [ ] move runtime path helpers there
- [ ] move launch script rendering there

Acceptance:
- `repl_host_v1_3` no longer owns cloud-init/mounts/layout string generation
- launch assets are testable without running the REPL

## Phase 3: Orchestrator and Blocking Readiness

Goal:
- create a typed lifecycle API suitable for agents

Tasks:
- [ ] add `libs/vmm/src/orchestrator.rs`
- [ ] define `LaunchArtifacts`, `VmHandle`, `ShutdownReport`
- [ ] add `prepare()`
- [ ] add `launch()`
- [ ] add `launch_and_wait()`
- [ ] add explicit readiness gates:
  - [ ] API socket ready
  - [ ] guestfs connected
  - [ ] SSH bridge connected
  - [ ] exec-ready probe

Acceptance:
- a caller can block until a guest is actually usable
- launch failures identify which readiness stage timed out or failed

## Phase 4: GuestFS and SSH Bridge Lifecycle Extraction

Goal:
- remove subsystem lifecycle wiring from the example

Tasks:
- [ ] add `libs/vmm/src/guestfs.rs`
- [ ] move guest provisioning and mount attachment there
- [ ] move guest listener spawn logic there
- [ ] move SSH bridge accept/register lifecycle out of `repl_host.rs`
- [ ] unify guest launch/SSH handle/vnet/VFS state into one library-owned handle

Acceptance:
- `repl_host_v1_3` does not manually coordinate SSH bridge and VFS listener state
- guest lifecycle state is tracked by the library, not by ad hoc REPL maps

## Phase 5: Automatic Guest Provisioning From SSH Principal

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

## Phase 6: Validation and Agent Harness Mode

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

## Phase 7: Polish and Hardening

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

## Phase 8: Guest Reporting and Metrics

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
