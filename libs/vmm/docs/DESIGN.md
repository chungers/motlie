# motlie-vmm: Reusable VM Orchestration Extracted from Proven Examples

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-12 | @codex-vmm | Refresh DESIGN from current merged reality: `v1.4` harness and PR #159 auto-provisioning are already proven, and the remaining work is reusable harness-core extraction, typed validation profiles, and the standard guest-path follow-up |
| 2026-04-09 | @codex | Rescope the post-merge harness direction: preserve `examples/v1.4/harness` as the historical origin and first consumer, but extract reusable scenario/validation infrastructure into `libs/vmm` without rewriting away the existing `v1.4` harness artifacts |
| 2026-04-08 | @codex | Address PR 140 review drift: remove the dead `VmBackend` / `BackendSet` transitional story from the design, update `GuestSpec` / `PreparedGuest` / shutdown snippets to match code, and record the typed `OverlaySize` plus namespace-sensitive socket-path allocation details |
| 2026-04-08 | @codex | Make the harness terminal-state engine switchable, adopt `shadow-terminal` as the default high-fidelity backend for PTY/TUI validation, keep `vt100` as an explicit fallback backend, and keep PNG/GIF/movie generation out of scope for `v1.4` |
| 2026-04-08 | @codex | Add the PTY export design decision: keep NDJSON transcript plus VTE screen JSON as canonical validation artifacts, add asciicast export for portable replay/interchange, and explicitly defer PNG/GIF/movie generation as out of scope for `v1.4` |
| 2026-04-08 | @codex | Replace the temporary 7-slot network allocator with a dedicated slot-derived allocation design section and public API: `Ipv4Subnet`, `Ipv4SubnetPool`, computed capacity, harness-exposed allocator config, and PTY/VTE scenario-driver direction |
| 2026-04-07 | @codex | Complete the remaining observability/result slice: `VmObservability` now exposes typed run-bundle metadata and capture paths, `harness_v1_4` persists internal result and PTY transcript artifacts, result JSON now carries structured failure classification, and PTY output is hardened into a stable evidence block |
| 2026-04-07 | @codex | Implement the first concrete Phase 4/5 slice in code: `observability.rs`, `VmHandle::observability()`, and `harness_v1_4 --result-json ...` for machine-readable `smoke` results; PTY result capture still needs hardening |
| 2026-04-07 | @codex | Lock the harness direction: `examples/v1.4/harness` is the future primary driver over the `libs/vmm` API, with scripted scenarios, interactive/manual mode, PTY/session control via `VmHandle`, and transcript/log capture; `repl_host_v1_4` remains transitional only |
| 2026-04-07 | @codex | Replace the intermediate VM-backend injection with reviewed `Runtime` injection so `orchestrator.rs` now takes composed hypervisor/filesystem/network/control-plane backing rather than importing Motlie implementation modules directly |
| 2026-04-07 | @codex | Complete the first usable `v1.4` lifecycle API: library-owned guestfs, SSH bridge, `VmHandle::exec(...)`, rootless harness validation, and child-handle-based shutdown/readiness instead of raw `/proc` polling |
| 2026-04-07 | @codex | Finish Phase 1/2 convergence in code and start Phase 3 with `PrepareRequest`, `PreparedGuest`, `VmHandle`, `backend/mod.rs`, and the first `backend::ch::shell::ChShellBackend` boot path |
| 2026-04-07 | @codex | Record the Cloud Hypervisor v44.0 internal Rust API analysis and tighten the reviewed layering around `GuestResources`, `GuestStorage`, and `BootArtifacts` below top-layer guest intent |
| 2026-04-07 | @codex | Tighten the reviewed `v1.4` API shape around `GuestUser`, `GuestSshAccess`, explicit CA-issued guest SSH credentials, and `boot()` plus `VmHandle::ready(...)` |
| 2026-04-07 | @codex | Start Phase 2 extraction in `libs/vmm/src/artifacts.rs` and record it as the owning module for rendered boot/runtime artifacts |
| 2026-04-07 | @codex | Add `libs/vmm/docs/API.md` as the running API review surface for `v1.4` extraction work |
| 2026-04-07 | @codex | Add a `v1.4` embedded-image / union-binary phase: prototype bundling an opinionated guest image into the harness ELF and booting from memfd-backed artifacts |
| 2026-04-07 | @codex | Insert an explicit `v1.4` programmatic harness bootstrap phase after lifecycle extraction so later phases build on a stable non-REPL substrate |
| 2026-04-07 | @codex | Add a `v1.4` automatic guest provisioning phase driven by incoming SSH principals and document the library-owned CID/IP/MAC allocation story |
| 2026-04-07 | @codex | Add a `v1.4` observability/reporting phase: combine Cloud Hypervisor host-side API/event data with guest-side SSH probes for CPU/memory/disk/network reporting |
| 2026-04-07 | @codex | Start `v1.4` as the library-extraction line: keep `v1.3` frozen for comparison, require a side-by-side `v1.4` namespace, and define the thin-harness target for `repl_host_v1_4` |
| 2026-04-06 | @codex | Reframe DESIGN around `examples/v1.3` as the active proving ground; add stable harness direction, blocking readiness requirements, and extraction targets for reusable orchestration APIs |
| 2026-04-05 | @claude-vmm | SSH proxy transport: vsock (AF_VSOCK) replaces TAP for ingress, completing fully-userspace stack; update NFR-1, FR-6, data flow |
| 2026-04-04 | @claude-vmm | Fold SSH proxy as programmatic guest control plane into DESIGN; add FR/NFR sections; update status to reflect v1.2 validation; resolve open questions |
| 2026-04-03 | @codex | Initial DESIGN for `libs/vmm`: capture the post-`v1.2` extraction target for reusable VM orchestration code |

## Status

The earlier `v1.2` lineage validated the composed subsystem flow:
guest launch composition, dual-network topology, cloud-init rendering,
VFS mount wiring, and deterministic shutdown.

`libs/vmm/examples/v1.3` is now the validated comparison baseline for the
VMM-owned harness. The current `v1.3` checkpoint has already proven:

- a runnable `motlie-vmm` example harness
- vsock-backed SSH proxy ingress on `localhost:2222`
- programmatic exec over the same SSH proxy path
- deterministic API -> SIGTERM -> SIGKILL shutdown fallback
- image/runtime conventions for agent-state, SSH CA injection, and guest boot

The active next step is `v1.4`:

- fork `examples/v1.3` into `examples/v1.4`
- move reusable lifecycle/readiness/orchestration logic into `libs/vmm/src`
- keep `v1.3` unchanged for comparison and regression analysis
- refactor `repl_host_v1_4` into a thin operator shell over reusable library
  services
- establish a programmatic `v1.4` harness after lifecycle extraction so later
  phases can be developed and tested against a stable non-REPL entrypoint
- prototype a distributable single-binary `v1.4` mode that embeds a curated
  guest image in the harness ELF and boots from memfd-backed artifacts
- move all `v1.4` bins, scripts, and runtime assets onto a distinct namespace
  so `v1.3` and `v1.4` can run side by side
- principal-driven guest auto-provisioning is now part of the merged `v1.4` line
- the next remaining work is to extract reusable harness and validation core
  into `libs/vmm`
- the next API-model follow-up after that extraction is guest-shape cleanup
  around `VmSpec` plus a simple standard Cloud Hypervisor guest path
- add a reporting layer that can answer both host-visible and guest-visible
  health/metrics questions during automated runs

Current `v1.4` implementation status:

- the reviewed Phase 1 and Phase 2 API surface is converged in code
- `Phase 3` now has a working lifecycle implementation:
  - contract in `backend/mod.rs`
  - implementation in `backend/ch/shell.rs`
  - placeholder vertical slices in `backend/motlie/` and `backend/vz/`
  - reviewed runtime injection via:
    - `Runtime`
    - `HypervisorBacking`
    - `FilesystemBacking`
    - `NetworkBacking`
    - `ControlPlaneBacking`
  - `orchestrator.rs`
  - `ChShellBackend`
  - `prepare()`
  - `boot()`
  - `LifecycleServices`
  - `VmHandle::ready(...)`
  - `VmHandle::exec(...)`
  - `VmHandle::shutdown()`
- `examples/v1.4/repl_host.rs` now exists as a thin harness over those library
  services
- `libs/vmm/src/guestfs.rs` now owns guestfs provisioning, mount attachment,
  and the guest listener spawn loop
- `examples/v1.4/harness/main.rs` now exists as a rootless automation harness
  that validates the library-owned lifecycle API end to end
- `VmHandle` now exposes a first PTY/session control surface for harness-driven
  interactive validation:
  - `VmHandle::open_pty(...)`
  - `GuestPtySession::send(...)`
  - `GuestPtySession::send_line(...)`
  - `GuestPtySession::resize(...)`
  - `GuestPtySession::read_for(...)`
  - `GuestPtySession::read_until_contains(...)`
  - `GuestPtySession::transcript()`
- `VmHandle` now exposes a first library-owned observability surface:
  - `VmHandle::observability()`
  - runtime/log/socket roots
  - active filesystem/network/control-plane backing identity
  - typed run-bundle metadata and standard capture paths for harness artifacts
- `harness_v1_4` now supports first-pass machine-readable result output:
  - `--result-json <path>`
  - structured `smoke` scenario results
  - named checks plus `VmObservability`
  - stable success/failure status and classified error records for agents/CI
  - PTY scenario evidence plus persisted transcript capture under the run bundle
  - PTY export as asciicast for portable replay in CLI/web viewers
- `ChShellBackend` now tracks the spawned child process directly in its
  backend-specific module so readiness and shutdown use real process state
  rather than `/proc` zombie heuristics
- generic orchestrator code no longer imports CH backend implementation
  modules directly for VM boot/shutdown dispatch
- generic orchestrator code no longer imports Motlie guestfs, userspace vnet,
  or SSH-bridge implementation modules directly either; those are now reached
  through reviewed `Runtime` composition
- the next remaining API-model step is guest-shape cleanup around `VmSpec`
  plus the simple Cloud Hypervisor “hello world” example
- the next harness step is direct:
  - extract reusable scenario, validation, and result infrastructure from
    `examples/v1.4/harness/` into `libs/vmm`
  - keep `examples/v1.4/harness/` as the concrete operator-facing harness that
    consumes that extracted core
  so the same harness machinery can serve:
  - scripted scenarios
  - coding-agent experimentation
  - human manual validation
  - future non-`v1.4` harness consumers
- `examples/v1.4/build-guest.sh` and `examples/v1.4/launch-ch.sh` exist under
  the `motlie-vmm-v14-*` namespace
- `RuntimeNamespace` now owns the core runtime-environment rules used by the
  harness and REPL:
  - root resolution from `MOTLIE_VMM_ROOT` or the platform temp dir
  - per-process namespace generation
  - guest vsock service socket naming
- `examples/v1.4/harness` and `examples/v1.4/repl_host_v1_4` now accept
  `--root <dir>` so live instances do not depend on a hardcoded `/tmp`
  root and can be isolated under caller-selected host directories

## Desired Outcome

`v1.4` should prove two end states:

1. Motlie-backed guest operation works end to end.
   - `backend::motlie::*` guest backing should support a real runnable
     harness plus test scripts that validate guest behavior.
   - This is the path proven today by the `v1.4` rootless harness using the
     Motlie guestfs / vnet / SSH-proxy stack.

2. The `motlie-vmm` API is also usable as a thin, simple abstraction over a
   simple standard Cloud Hypervisor guest.
   - A caller should be able to boot a guest with ordinary hypervisor-managed
     networking and storage, with no Motlie-specific guestfs or userspace vnet
     providers involved.
   - This should be demonstrated first by a small CH “hello world” example
     that boots a guest through the same `prepare()` / `boot()` / `ready()` /
     `shutdown()` lifecycle API without using Motlie guest backing providers.
   - The same portable slice should later map to `backend::vz::*`.

3. `examples/v1.4/harness` is the concrete operator-facing harness for the
   merged `v1.4` line.
   - It should keep supporting repeatable scenarios and ad-hoc/manual
     operation.
   - It should keep driving PTY sessions, capturing transcripts, and bundling
     useful run artifacts such as launch logs and serial logs.
   - The old `repl_host_v1_4` is transitional and should not accumulate unique
     control-plane logic.

4. Reusable harness core should live in `libs/vmm`.
   - Reusable scenario definitions, step/result types, validation profiles,
     and driver/result plumbing should move into the library.
   - `examples/v1.4/harness` should consume that extracted core.
   - `v1.4`-specific guest/image setup, guest catalog, runbook docs, and
     operator-facing UX should stay in `examples/v1.4`.

## Cloud Hypervisor API Analysis

Cloud Hypervisor `v44.0.0` already exposes an internal Rust API surface that is
more structured than the CLI wrapper used by the current examples.

Important local source points used for this analysis:

- [`src/main.rs`](/tmp/cloud-hypervisor-v44/src/main.rs)
- [`vmm/src/lib.rs`](/tmp/cloud-hypervisor-v44/vmm/src/lib.rs)
- [`vmm/src/vm_config.rs`](/tmp/cloud-hypervisor-v44/vmm/src/vm_config.rs)
- [`vmm/src/api/mod.rs`](/tmp/cloud-hypervisor-v44/vmm/src/api/mod.rs)

The important finding is that the CLI is just a front-end over the `vmm` crate:

1. build a `VmConfig`
2. start the event monitor thread separately when configured
3. start the VMM thread through `vmm::start_vmm_thread(...)`
4. send `VmCreate(Box<VmConfig>)`
5. send `VmBoot`

This matters for `v1.4` because it means the library should be designed so our
typed guest inputs can be translated almost mechanically into CH's internal
`VmConfig`, instead of baking CLI-string construction into the long-term API.

### CH-Shaped Inputs

The CH `VmConfig` surface cleanly separates:

- `CpusConfig`
- `MemoryConfig`
- `PayloadConfig`
- `DiskConfig`
- `NetConfig`
- `FsConfig`
- `VsockConfig`
- `RngConfig`
- `BalloonConfig`
- `PlatformConfig`

### Design Consequence

The reviewed `v1.4` API should therefore be layered:

- top layer: guest intent
  - guest user
  - SSH access policy
  - software profile
- portable VM slice
  - `VmSpec`
  - `GuestResources`
  - `GuestStorage`
  - `BootArtifacts`
- runtime composition
  - `Runtime`
  - `HypervisorBacking`
  - `FilesystemBacking`
  - `NetworkBacking`
  - `ControlPlaneBacking`
- backend realization
  - `backend::ch::*`
  - `backend::motlie::*`
  - `backend::vz::*`

The critical constraint is:

- top-layer Motlie concepts such as `GuestUser` and `GuestSshAccess` should
  stay above the CH adapter
- CH-facing configuration should be modeled explicitly enough that a future
  `to_ch_vm_config(...)` step is straightforward
- product/example-specific guest catalogs and operator UX should stay above
  `libs/vmm`
- reusable harness-core infrastructure may move into `libs/vmm` once proven in
  `examples/v1.4`
- the library should expose both:
  - lifecycle, exec, PTY, and observability primitives for any caller
  - reusable scenario/validation core that later harnesses can consume without
    cloning the `v1.4` engine

## Harness Direction

`examples/v1.4/harness/` is the concrete `v1.4` operator harness.

It must keep supporting:

- scripted scenario mode
  - action/expectation pairs for repeatable regression tests
- interactive/manual mode
  - human-driven exploratory testing
  - coding-agent-driven ad-hoc reproduction and debugging

The extraction rule is:

- move reusable harness-core modules into `libs/vmm`
- keep `examples/v1.4/harness/` as the `v1.4`-specific harness binary and UX
- keep `v1.4`-specific guest/image/runbook content in `examples/v1.4`

Both the original `v1.4` harness and the extracted harness core must use the
same `libs/vmm` lifecycle/control-plane APIs:

- `prepare(...)`
- `boot(...)`
- `VmHandle::ready(...)`
- `VmHandle::exec(...)`
- `VmHandle::open_pty(...)`
- `VmHandle::shutdown()`

The harness should collect and preserve enough observability for debugging:

- launch/boot log
- serial/console log
- generated guest artifacts
- PTY transcript
- harness-side stdout/stderr summaries

The first concrete implementation slice for this is now:

- library-owned `VmHandle::observability()`
- harness `--result-json <path>` output for the `smoke` scenario
- typed run-bundle metadata and capture paths under `VmObservability`
- persisted PTY transcript and internal result artifacts under the bundle root

The terminal/reporting contract is now:

- rendered terminal state / VTE capture is in scope and part of the harness
  contract
- the terminal-state engine is switchable inside the harness line
  - `shadow` is the default high-fidelity backend for PTY/TUI validation
  - `vt100` remains available as an explicit fallback/backend-comparison mode
- the key decision is to keep the terminal backend boundary inside the harness
  layer rather than inside the lower-level VM lifecycle primitives, because
  the harness line owns the live PTY stream and artifact persistence
- chosen `v1.4` recording direction: asciicast export is in scope as a
  portable text/timing replay format layered on top of the canonical NDJSON
  transcript and rendered screen JSON
- PNG, GIF, MP4, or other rendered movie artifacts are out of scope for
  `v1.4`; those are human-review outputs, not the primary validation contract
  for future agents

The practical reason for the `shadow` default is Codex/TUI fidelity: the
`shadow-terminal`/WezTerm-backed renderer produces a clean enough rendered
screen for alternate-screen startup validation, while the lighter `vt100`
backend remains useful as a cheap compatibility parser but is not the default
for agent-facing TUI assertions.

This is explicitly meant to replace the old split between:

- standalone REPL for ad-hoc use
- separate smoke binaries/scripts for automation

The harness line should become the single coherent driver over the same
underlying library APIs.

The extracted `libs/vmm` harness core should sit underneath
`examples/v1.4/harness`, while `v1.4`-specific guest, image, and operator
artifacts stay in `examples/v1.4`.

One specific design correction from this analysis:

- writable overlay sizing should not live inside generic compute/memory
  resources
- it belongs to storage/image modeling instead

That is why the reviewed API now separates:

- `GuestResources`
- `GuestStorage`
- `BootArtifacts`

### Backend consequence

The library should also define a backend seam that is generic enough for:

- Cloud Hypervisor shell-driven launch
- Cloud Hypervisor fork/exec launch
- Cloud Hypervisor in-process `start_vmm_thread(...)`
- future macOS `vz` support

Even though `ch_shell`, `ch_fork_exec`, and `ch_vmm_thread` all target Cloud
Hypervisor, they should still be treated as distinct backends because their:

- boot mechanics
- shutdown mechanics
- cleanup behavior
- control-plane attachment
- isolation/fault boundaries

are materially different.

This backend seam should use enum dispatch, not dynamic discovery or plugin
loading, because all supported backends are expected to be known and
implemented in-tree.

The generic orchestrator should therefore depend on an injected reviewed
`Runtime`, not on concrete backend modules directly. The current implementation
now does this for the active `v1.4` path: VM boot, filesystem backing,
userspace network backing, and the SSH bridge are all composed outside
`orchestrator.rs` and injected as one runtime composition.

### Anti-Drift Rule: One Backend Namespace, Separate Composition Axes

To avoid future architectural drift, `v1.4` should follow this rule:

- use one implementation namespace under `libs/vmm/src/backend/`
- but do not collapse all backend concepts into one exclusive runtime choice

In particular, the design must keep these ideas separate:

- hypervisor backing
  - example: `backend::ch::shell`
  - future: `backend::ch::fork_exec`, `backend::ch::vmm_thread`,
    `backend::vz::*`
- guest capability backends
  - example: `backend::motlie::vfs`
  - example: `backend::motlie::vnet`
  - example: `backend::motlie::ssh_proxy`

The reason is that these are composable, not mutually exclusive. A valid guest
runtime may be:

- Cloud Hypervisor shell + hypervisor-managed filesystem + hypervisor-managed
  networking
- Cloud Hypervisor shell + Motlie VFS backing
- Cloud Hypervisor shell + Motlie VFS backing + Motlie userspace vnet backing
- future Apple Virtualization + native networking + optional Motlie guest
  capability backends when supported

So the crate should group all implementations under `backend/` for coherence,
build flags, and vertical slices, but the API and orchestrator must preserve
separate composition axes for the reviewed:

- `Runtime`
  - `hypervisor: HypervisorBacking`
  - `filesystem: FilesystemBacking`
  - `network: NetworkBacking`
  - `control_plane: ControlPlaneBacking`

This is intentionally more explicit than a single `backend = ...` choice. The
design should tolerate duplication between backend families rather than forcing
an early shared base layer that will later fork.

## Problem Statement

The current example lineage (`v1`, `v1.1`, `v1.2`, and now `v1.3`) contains
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

1. we freeze the wrong API before the example flow proves the workflow
2. we blur crate boundaries by pushing VM orchestration into `motlie-vfs` or
   `motlie-vnet`

We need a dedicated home for reusable VM orchestration once the example flow is
validated, while keeping early experimentation in examples where it is easy to
change. In `v1.3`, the remaining gap is no longer "can this work?" but
"can we turn the working harness into a stable, typed API that agents and
future CLIs can rely on without racing guest readiness or rebuilding ad hoc
state?"

## Goals

- Provide a dedicated library for host-side VM orchestration and launch
  composition.
- Extract reusable code only after it is proven in the current example line.
- Keep device-specific logic in the device crates:
  - `motlie-vfs` owns filesystem serving and guest mount transport
  - `motlie-vnet` owns outbound guest networking and the vhost-user-net backend
- Centralize Cloud Hypervisor launch construction and per-guest runtime layout.
- Centralize guest boot asset generation such as cloud-init and generated
  `mounts.yaml`.
- Make multi-guest orchestration and per-guest lifecycle management reusable.
- Provide a host-side SSH proxy (russh) that replaces TAP-based ingress,
  completing the fully-userspace stack and enabling programmatic guest
  command execution for automated testing.
- Let future examples become thin wiring layers over library code rather than
  carrying large shell-script control planes.
- Add a stable reporting surface for guest lifecycle, resource use, and device
  counters so future Motlie work can debug regressions without manual SSH-only
  inspection.
- Add a stable auto-provisioning path so new guest principals can be created on
  first use without embedding more lifecycle/allocation policy into the example
  REPL.
- Add a programmatic harness substrate that can be used to iteratively build
  and validate later `v1.4` phases without coupling new work to REPL text/UI.

## API Review

The evolving review surface for the extracted library API lives in:

- [API.md](/tmp/vmm-v1.4/libs/vmm/docs/API.md)

This is where Phase 1+ type/module surfaces and example usage should be
documented as extraction proceeds.

Current extraction checkpoints:

- Phase 1: `spec.rs`, `network.rs`, `network_alloc.rs`
- Phase 2: `artifacts.rs`
- Phase 3 initial slice: `backend.rs`, `orchestrator.rs`

Current convergence status:

- Phase 1 reviewed naming is implemented in code
- Phase 2 render APIs now consume software, storage, and boot-artifact inputs
- Phase 3 has started with:
  - `PrepareRequest`
  - `PreparedGuest`
  - `VmHandle`
  - `BackendKind`
  - `VmBackendCapabilities`
  - `BackendHandle`
  - `ChShellBackend`
  - `prepare()`
  - `boot()`
  - `VmHandle::ready(...)`
  - `VmHandle::shutdown()`
- current readiness coverage is intentionally narrow:
  - API socket readiness is implemented
  - guestfs / SSH bridge / exec-ready gates remain follow-up work in Phase 3
- Explore a one-binary distribution model where the host harness embeds an
  opinionated guest image and can boot it without requiring a separate image
  bundle on disk.

## Non-Goals

- Defining a general-purpose hypervisor abstraction across VMMs.
- Replacing `cloud-hypervisor` with a pluggable backend in v1.
- Moving `motlie-vfs` protocol logic into `libs/vmm`.
- Moving `motlie-vnet` backend logic into `libs/vmm`.
- Freezing the final public Rust API before the harness flow validates the operator flow.
- Eliminating example scripts immediately; scripts may remain as thin wrappers.

## Functional Requirements

### FR-1: Guest Lifecycle Orchestration

The library must support prepare → boot → running → shutdown → cleanup
for one or more guest VMs. Each lifecycle transition must be explicit and
deterministic. The orchestrator manages per-guest state and coordinates
subsystem startup/teardown order.

For harness-grade automation, guest startup must support both:

- asynchronous fire-and-forget startup for operator workflows
- explicit blocking readiness for agents and tests

The ready state must be explicit rather than inferred from opportunistic SSH
success.

### FR-2: Launch Composition

Compose Cloud Hypervisor launch arguments from typed guest, network, and
subsystem configuration. This includes:

- admin ingress vs egress network selection
- per-mode CH device arguments (vhost-user sockets, shared memory, TAP)
- NIC MAC assignment and guest-visible route ownership
- shared-memory region sizing for vhost-user devices

### FR-3: Boot Asset Generation

Render guest boot assets from typed state rather than string templates:

- cloud-init `user-data` (user account, SSH keys, boot services)
- cloud-init `meta-data` (instance-id, hostname)
- optional `network-config` (egress NIC DHCP matching on stable MAC)
- generated `mounts.yaml` (VFS mount tags → guest paths)
- optional software profile inputs (for example `vim`, `gh`) that can be
  rendered into boot-time install behavior during development

Important design rule:

- software customization should be expressed as typed intent
- not hardwired to one implementation strategy

That allows the same requested software set to map to:

1. cloud-init package installation in the development flow
2. baked image composition in the later union-binary phase

### FR-4: Subsystem Wiring

Start and stop `motlie-vnet` and coordinate `motlie-vfs` host-side
provisioning with guest launch. Cleanup order must be deterministic:
guests shut down before subsystem backends are torn down.

### FR-5: Runtime Layout

Deterministic per-guest directory structure:

- runtime directory naming and lifecycle
- overlay image paths
- cloud-init seed paths
- per-guest vsock, API, and vnet sockets
- serial/log output locations

### FR-6: SSH Proxy — Host-Side Ingress and Programmatic Control Plane

An in-process SSH server (russh) that serves two roles:

1. **User-facing SSH ingress.** Replaces TAP-based admin SSH path.
   Users connect via `ssh -p <port> <guest>@localhost`. The proxy
   extracts the username as the guest identity, ensures the VM is
   running, signs an ephemeral CA cert (Ed25519, 60s TTL), and bridges
   the external SSH channel to the guest's stock `openssh-server`.

2. **Programmatic guest command execution.** The host orchestrator
   (or an automated test harness) uses the same russh client path to
   open SSH channels and execute commands inside the guest without
   human intervention:

   ```rust
   // Host-side automated validation — no human, no PTY needed:
   let output = orchestrator.exec(&guest, "curl -s -o /dev/null -w '%{http_code}' https://example.com").await?;
   assert_eq!(output.stdout.trim(), "200");

   let output = orchestrator.exec(&guest, "readlink ~/.codex").await?;
   assert_eq!(output.stdout.trim(), "/agent-state/codex");
   ```

   This enables fully agent-driven testing of guest networking,
   filesystem behavior, and subsystem integration — no human in the
   loop, suitable for CI.

The same proxy path must be good enough for:

- human interactive shell access
- agent-driven command execution
- future structured readiness probes and health checks

### FR-8: Guest Reporting and Metrics

The library must provide a reusable reporting path for VM status and resource
usage that supports both:

- operator debugging in the REPL
- machine-readable snapshots during automated tests

The reporting surface should intentionally combine two sources:

1. **Cloud Hypervisor host-side reporting**
   - API socket exposed by `--api-socket`
   - event stream exposed by `--event-monitor`
   - API calls including:
     - `/api/v1/vm.info`
     - `/api/v1/vm.counters`

2. **Guest-side reporting over the existing SSH exec path**
   - CPU utilization probes
   - guest memory usage probes
   - disk/filesystem usage probes
   - process/service health probes
   - outbound connectivity checks

This split is deliberate. Cloud Hypervisor can report VMM-visible state and
device counters, but it does not replace guest-OS introspection for metrics
like guest-used memory, per-process CPU, or filesystem occupancy.

The intended reusable library surface is something like:

```rust
pub struct VmReport {
    pub lifecycle: VmLifecycleReport,
    pub host: VmHostReport,
    pub guest: Option<GuestProbeReport>,
}
```

Where:

- `VmLifecycleReport` captures readiness/shutdown state
- `VmHostReport` captures CH-visible state, counters, and event snapshots
- `GuestProbeReport` captures SSH-collected guest metrics and health

### FR-9: Automatic Guest Provisioning From Incoming SSH Principals

The library must support creating a new guest on first contact when the SSH
proxy receives a principal that does not yet map to a known guest.

Example:

- `alice@localhost` -> resolves to existing `alice` guest if already provisioned
- `bob@localhost` -> resolves to existing `bob` guest if already provisioned
- `jane@localhost` -> allocates and provisions a new `jane` guest if missing
- `mike@localhost` -> allocates and provisions a new `mike` guest if missing

This must be library-owned behavior, not ad hoc REPL glue. The orchestrator
should provide a flow like:

```rust
let handle = orchestrator.ensure_guest_for_principal("jane").await?;
```

The resulting behavior should be:

1. look up existing guest by principal
2. if missing, allocate stable guest identity/resources
3. provision guestfs/runtime paths
4. boot and wait until ready
5. continue with the SSH session against that guest

### FR-10: Programmatic Harness Bootstrap

Before later `v1.4` phases such as auto-provisioning, reporting, and richer
automation, the library must expose a stable non-interactive harness surface
that can drive the reusable lifecycle APIs directly.

This should be owned by `examples/v1.4`, but it must be library-first rather
than REPL-first.

The minimum useful surface is:

- `boot`
- `handle.ready(...)`
- `exec`
- `handle.shutdown()`
- machine-readable status/result output

The purpose of this phase is not to replace the human REPL. It is to create a
stable development substrate so subsequent `v1.4` feature phases can be built,
tested, and debugged without depending on interactive prompt behavior.

This feature belongs after lifecycle, guestfs, and SSH bridge extraction
because it depends on all three being library-owned already.

### FR-11: Embedded Image / Union Binary Prototype

The `v1.4` line should include an explicit prototype phase for building a
single distributable binary that embeds an opinionated guest image payload in
its ELF `.rodata` section.

The intended operator experience is:

- one host binary
- userspace-only runtime
- `kvm` group membership as the main host requirement
- no separate guest image bundle required at runtime

The prototype should use a special build flag to produce this combined artifact
and should prefer memfd-backed boot assets where Cloud Hypervisor supports
path-based handoff via `/proc/self/fd/...`.

The design should preserve two modes:

1. normal development/image-on-disk mode
2. special union-binary prototype mode

This phase is most appropriate after:

- **Phase 2**, when image/build artifact assembly is finally library-owned
- **Phase 5**, when the programmatic harness exists and can repeatedly validate
  the prototype without depending on the REPL

So image construction is stabilized enough by Phase 2, but the feature becomes
an efficient prototype target after the harness bootstrap in Phase 5.

**Auth model:**

- **Inbound (client → russh):** Localhost trust. russh binds
  `127.0.0.1` only. Both `auth_none` and `auth_publickey` accept
  unconditionally — the username is the only extracted value. If you
  can reach localhost, you already own the host-side credential and
  workspace directories. The SSH protocol is used as a session
  transport and identity carrier, not as an authentication gate.

- **Outbound (russh → guest sshd):** CA-based ephemeral certs. The
  daemon holds a user CA keypair in memory. On each connection it
  signs a throwaway Ed25519 cert with `principal=<username>` and
  60-second TTL. The guest image has the CA public key baked into
  `/etc/ssh/ca/user_ca.pub` and each VM's
  `/etc/ssh/auth_principals/root` contains only that VM's username.
  Even with a valid cert for "bob", you cannot reach alice's VM.

- **Scope limitation:** The localhost-only trust model is appropriate
  for single-user development hosts. Multi-tenant or network-exposed
  deployments would require real inbound authentication (e.g.
  publickey verification against an authorized keys source). That is
  out of scope for v1.

### FR-7: Automated Guest Validation

The orchestrator must expose a command-execution interface (`exec`) that
captures stdout, stderr, and exit code from commands run inside a guest.
This is the primitive that enables:

- CI-driven regression testing of the full guest stack
- Automated runbook execution (every step in the validated harness
  runbook becomes a programmatic assertion)
- Agent-driven guest provisioning without SSH shell sessions

Every step in the current manual harness runbook maps directly:

| Manual step | Programmatic equivalent |
|---|---|
| `ssh -p <proxy-port> alice@localhost` then `ip route` | `orchestrator.exec(&alice, "ip route")` |
| `curl -I https://example.com` | `orchestrator.exec(&alice, "curl ...")` → assert exit 0 |
| `readlink ~/.codex` | `orchestrator.exec(&alice, "readlink ~/.codex")` → assert `/agent-state/codex` |
| `sudo apt-get update` | `orchestrator.exec(&alice, "sudo apt-get update")` → assert exit 0 |

## Non-Functional Requirements

### NFR-1: Fully Userspace Operation

With the SSH proxy replacing TAP-based ingress, the entire host-side
stack runs without elevated privileges:

| Layer | Mechanism | Privilege |
|---|---|---|
| SSH ingress | russh on `127.0.0.1:<port>` | None (unprivileged port) |
| VM execution | KVM via `/dev/kvm` | `kvm` group membership |
| Guest networking (egress) | motlie-vnet (vhost-user + libslirp) | None — userspace sockets |
| Guest filesystems | motlie-vfs over vsock | None |
| Host↔guest transport | `/dev/vhost-vsock` | `kvm` group membership |
| Guest networking (ingress) | SSH proxy (replaces TAP) | None |

No `CAP_NET_ADMIN`, no `sudo`, no `setcap`, no `iptables`, no TAP
devices, no network namespaces, no host network interface modifications.

### NFR-2: No Guest-Side Dependencies for Control

Programmatic `exec` uses standard SSH `channel_exec` — the guest needs
only stock `openssh-server`, not a custom control agent. This keeps the
guest image generic and avoids coupling the test harness to a bespoke
guest-side protocol.

### NFR-3: Cloud Hypervisor-Specific in v1

`libs/vmm` targets Cloud Hypervisor directly in v1. No hypervisor
abstraction trait, no pluggable backend. Internal module boundaries
should be clean enough that a future backend could be introduced, but
v1 does not design for that or expose seams for it.

<!-- @claude-vmm 2026-04-04 — Resolves open question "CH-specific or CH-first
     with seams?" The non-goals already exclude pluggable backends in v1.
     A trait adds complexity with zero current consumers. -->

## Design Principles

### 1. Prove First, Extract Second

`examples/v1.3` is the place to prove:

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

That rule also applies to guest identity allocation. The SSH principal to guest
mapping, CID selection, MAC selection, and runtime namespace derivation must be
typed library state, not implicit shell naming conventions.

It also applies to harness execution. Future phases should be proven against a
typed programmatic harness surface before they are considered stable enough to
become part of the long-lived `v1.4` flow.

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
- SSH proxy (russh): user-facing ingress and programmatic guest exec
- SSH CA: in-memory keypair, ephemeral cert signing for guest auth
- automated guest validation primitives (`exec` → stdout/stderr/exit)

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
    ssh.rs          # russh server + client, channel bridging, exec
    ca.rs           # SSH CA keypair, ephemeral cert signing
    orchestrator.rs # high-level "prepare, start, stop, exec" flow
```

This is a target shape, not a commitment to specific filenames.

`ssh.rs` and `ca.rs` are the new modules supporting FR-6/FR-7. The SSH
proxy is an orchestration concern — it composes guest identity (from
`guest.rs`), VM lifecycle (from `orchestrator.rs`), and CA signing (from
`ca.rs`) into a single ingress/control-plane endpoint.

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

The reviewed API distinction is:

- `BootArtifacts` is the declarative boot-input model above the renderer
- `artifacts.rs` produces the concrete rendered files and paths below that API
  layer

The owning module for this work in `v1.4` is:

- [artifacts.rs](/tmp/vmm-v1.4/libs/vmm/src/artifacts.rs)

This is also the point where image/build artifact handling becomes stable
enough for the later union-binary prototype phase, even though the
programmatic harness bootstrap still needs to land before that prototype is a
good iterative development target.

### 3. Runtime Layout and Artifact Assembly

Expected extraction:

- runtime directory naming
- overlay image paths
- cloud-init seed paths
- per-guest sockets and API sockets
- launch-helper log/serial locations

This logic is already shaping into a reusable contract in the example lineage.

### 3a. Guest Allocation Policy and Identity

The `v1.4` line should make the guest identity allocator explicit and
library-owned. The current scaffold is in:

- [network_alloc.rs](/tmp/vmm-v1.4/libs/vmm/src/network_alloc.rs)

The intended policy is:

- each guest principal maps to one stable assignment for the lifetime of the
  harness process
- shutting a guest down does not lose its assignment
- reboot/restart reuses the same assignment
- a new guest principal consumes the next free slot
- exhaustion is a typed error, not silent wraparound/saturation

The assignment should include at least:

- guest slot
- vsock CID
- admin ingress subnet/IP pair when used
- admin MAC
- egress MAC
- vhost-user socket path
- runtime namespace roots

Current direction:

- `v1.3` continues using fixed example-owned provisioning for known guests
- `v1.4` introduces library-owned dynamic provisioning keyed by SSH principal
- the SSH proxy should call the orchestrator, not the REPL, to resolve or
  create guests

### 4. Subsystem Lifecycle Wiring

Expected extraction:

- start/stop `motlie-vnet`
- coordinate `motlie-vfs` host-side provisioning with guest launch
- ensure cleanup order is deterministic on shutdown

This code should become reusable once the composed `v1.2` flow is stable.

### 5. SSH Proxy and Programmatic Control Plane

This is a **new component**, not an extraction from v1.2. The design
originates from `docs/motlie-vmm.md` (§8 Daemon, §9 SSH Server).

The SSH proxy replaces TAP-based admin ingress with a fully-userspace
path and simultaneously provides the host with programmatic command
execution inside guests. The data flow:

```
 User / test harness
        │
        ▼
 russh server (127.0.0.1:2222, TCP)
        │ extract username → guest identity
        ▼
 orchestrator.ensure_vm(guest)
        │ boot VM if needed
        ▼
 ca.sign_ephemeral(guest)
        │ Ed25519, principal=guest, TTL=60s
        ▼
 VsockStream::connect(cid, 2222)         ← AF_VSOCK, not TCP/TAP
        │ host kernel routes via vhost-vsock
        ▼
 guest socat (vsock:2222 → TCP:localhost:22)
        │
        ▼
 russh::client::connect_stream(vsock_stream, cert)
        │ guest sshd validates CA + principal
        ▼
 channel bridge (interactive)    ─or─    channel exec (programmatic)
   pty_request + shell_request              exec("command") → ExecOutput
   data ↔ data bidirectional                stdout, stderr, exit_code
```

**Transport: vsock, not TAP.** The proxy reaches guest sshd over
AF_VSOCK — a direct host↔guest memory channel — rather than TCP over
a TAP NIC. The guest runs a `socat` systemd service that bridges vsock
port 2222 to TCP `localhost:22` (stock openssh-server). This choice:

- **Eliminates TAP and `CAP_NET_ADMIN`** for ingress entirely
- **Isolates ingress from egress** — if libslirp/vnet has issues, SSH
  still works (critical for debugging)
- **Reuses existing infrastructure** — vsock is already configured per
  guest for VFS mounts (`/dev/vhost-vsock` + CID)
- **Aligns with the product architecture** (`docs/motlie-vmm.md` §10)

For interactive sessions, the proxy bridges PTY, data, and
window-change events bidirectionally. For programmatic exec, it opens
a non-PTY channel, sends the command, and collects output — this is the
primitive that FR-7 requires for automated validation.

The SSH proxy is the component that **eliminates TAP** from the ingress
path. The only kernel interfaces the full stack requires
are `/dev/kvm` and `/dev/vhost-vsock` — both accessible via `kvm` group
membership, no capabilities or root needed.

## API Direction

The API should be small and orchestration-centered. These shapes are
derived from the proven example patterns (`GuestConfig`, `GuestRuntime`,
`AdminNet`/`EgressNet`, `render_cloud_init`, `render_launch_script`,
`ensure_vnet_backend`, `shutdown_guest`) and the SSH proxy design from
`docs/motlie-vmm.md`.

### Core types

```rust
pub struct GuestSpec {
    pub guest_id: String,
    pub hostname: String,
    pub socket_path: PathBuf,
    pub user: GuestUser,
    pub ssh: GuestSshAccess,
    pub mounts: Vec<GuestMountSpec>,
    pub software: SoftwareProfile,
    pub resources: GuestResources,
    pub storage: GuestStorage,
    pub boot: BootArtifacts,
}

pub struct GuestUser {
    pub name: String,
    pub uid: u32,
    pub gid: u32,
    pub home: PathBuf,
}

pub struct GuestSshAccess {
    pub principal: String,
    pub login_user: String,
}

pub struct GuestMountSpec {
    pub tag: String,
    pub guest_path: Option<PathBuf>,
    pub host_path: PathBuf,
}

pub struct SoftwareProfile {
    pub packages: Vec<String>,
}

pub struct GuestResources {
    pub boot_vcpus: u8,
    pub memory_mib: u32,
    pub max_vcpus: Option<u8>,
}

pub struct GuestStorage {
    pub overlay_size: OverlaySize,
}

pub struct OverlaySize(String);

pub struct BootArtifacts {
    pub kernel: PathBuf,
    pub initramfs: Option<PathBuf>,
    pub firmware: Option<PathBuf>,
    pub cmdline: Option<String>,
}

pub struct PreparedGuest {
    pub guest: GuestSpec,
    pub namespace: RuntimeNamespace,
    pub runtime_paths: GuestRuntimePaths,
    pub guest_socket_path: PathBuf,
    pub net_assignment: GuestNetAssignment,
    pub cloud_init: CloudInitArtifacts,
    pub launch_script: String,
    pub network_modes: NetworkModes,
    pub base_dir: PathBuf,
}

/// Handle to a running guest VM.
pub struct VmHandle {
    pub guest_id: String,
    pub pid: Option<u32>,
    pub runtime_paths: GuestRuntimePaths,
    pub net_assignment: GuestNetAssignment,
}

/// Explicit readiness stages for blocking launch.
pub enum ReadinessStage {
    LaunchSpawned,
    ApiSocketReady,
    GuestFsConnected,
    SshBridgeReady,
    ExecReady,
}

pub struct ReadinessPolicy {
    pub api_socket_timeout: Duration,
    pub guestfs_timeout: Duration,
    pub ssh_bridge_timeout: Duration,
    pub exec_ready_timeout: Duration,
}
```

Notes:

- `GuestResources` is intentionally CH-shaped for `v1.4` because the active
  proving path is still Cloud Hypervisor shell boot plus Motlie guest backing
  providers.
- The next convergence target remains a broader `VmSpec` / device model for
  backends that need GPU, NUMA, pinned-memory, or other non-CH resource
  concepts. `v1.4` should not guess that final shape prematurely.

### Orchestration

The reviewed orchestration surface should use `boot()` as the start verb and
hang readiness off the returned handle:

```rust
pub fn prepare(req: PrepareRequest) -> Result<PreparedGuest, OrchestratorError>;
pub async fn boot(
    prepared: PreparedGuest,
    services: LifecycleServices,
) -> Result<VmHandle, OrchestratorError>;

impl VmHandle {
    pub async fn ready(
        &self,
        policy: &ReadinessPolicy,
    ) -> Result<(), OrchestratorError>;

    pub async fn shutdown(&self)
        -> Result<ShutdownReport, OrchestratorError>;
}
```

This is preferred over proliferating variants such as `launch_and_wait()` or
`boot_and_wait()` because:

- there is one obvious start verb
- readiness is clearly a state transition on a running VM handle
- later harness and provisioning code can boot first and then decide whether,
  when, and how long to wait for readiness

### Backend surface

The reviewed backend surface should stay narrow and sit below
`PreparedGuest`/`VmHandle`.

```rust
pub enum BackendKind {
    ChShell,
    ChForkExec,
    ChVmmThread,
    Vz,
}

pub struct VmBackendCapabilities {
    pub same_process_vmm: bool,
    pub supports_api_socket: bool,
    pub supports_event_monitor: bool,
    pub supports_fd_handoff: bool,
    pub supports_memfd_boot_artifacts: bool,
    pub supports_guest_metrics: bool,
}

pub enum BackendHandle {
    ChShell(ChShellHandle),
}

impl ChShellBackend {
    pub fn kind(&self) -> BackendKind;
    pub fn capabilities(&self) -> VmBackendCapabilities;
    pub fn boot(&self, prepared: &PreparedGuest) -> Result<BackendHandle, BackendError>;
    pub fn shutdown(
        &self,
        handle: &BackendHandle,
    ) -> Result<BackendShutdownOutcome, BackendError>;
}
```

Important design rules:

- no dynamic dispatch or runtime-discovered backends
- use enum dispatch because the backend set is known in source
- do not introduce a shared `VmBackend` trait when enum dispatch already covers
  the reviewed backend set
- out-of-tree or plugin-style backend extension is not a `v1.4` goal; if the
  in-tree backend/provider count grows enough that enum dispatch becomes
  unmanageable, that is the point to revisit trait-object or registry-based
  dispatch in a later design round
- do not push readiness, SSH exec, validation, or guestfs semantics into the
  backend implementation boundary
- the first implementation should be `ChShellBackend`, which preserves the
  current `v1.3` shell/CLI semantics behind the new interface

That gives us:

- stable top-level Motlie API
- uniform backend contract
- room to add `Vz` later without redesigning orchestration semantics

### Future Cloud Hypervisor integration approach

The intended future integration point is:

```rust
fn to_ch_vm_config(
    prepared: &PreparedGuest,
    ch: &ChVmOptions,
) -> Result<cloud_hypervisor_vmm::vm_config::VmConfig, ChConfigError>;
```

Where:

- `PreparedGuest` carries the reviewed Motlie-layer state
- `ChVmOptions` carries CH-specific tuning we do not want to bake into the
  generic API too early
- the adapter owns translation into CH's exact `VmConfig`

The orchestrator should eventually be able to swap:

- current shell/CLI invocation
- future in-process CH `VmConfig` + `start_vmm_thread(...)` path

without forcing a redesign of the upper Motlie API.

The reviewed transition path is:

1. `ChShellBackend`
2. `ChForkExecBackend`
3. `ChVmmThreadBackend`
4. future `VzBackend`

The initial implementation should stay simple and preserve current working
behavior by starting with `ChShellBackend`.

### SSH user/access binding

The binding between guest OS user state and SSH principal routing must be
explicit in the API.

```rust
pub struct IssuedGuestSshCredentials {
    pub principal: String,
    pub login_user: String,
    pub private_key_openssh: String,
    pub certificate_openssh: String,
}

impl SshCa {
    pub fn issue_guest_ssh_credentials(
        &self,
        user: &GuestUser,
        access: &GuestSshAccess,
    ) -> Result<IssuedGuestSshCredentials, CaError>;
}
```

This is the reviewed binding point where:

- `GuestUser` models the in-guest account
- `GuestSshAccess` models principal/login-user policy
- the CA issues the ephemeral credential material that binds the two

### Usage: automated guest validation (FR-7)

```rust
let spec = GuestSpec {
    guest_id: "alice".into(),
    hostname: "motlie-alice".into(),
    socket_path: "/tmp/motlie-vmm-v14-alice.vsock_5000".into(),
    user: GuestUser {
        name: "alice".into(),
        uid: 1000,
        gid: 1000,
        home: "/home/alice".into(),
    },
    ssh: GuestSshAccess {
        principal: "alice".into(),
        login_user: "alice".into(),
    },
    mounts: vec![
        GuestMountSpec { tag: "alice-home".into(), guest_path: Some("/home/alice".into()), host_path: "/tmp/demo/alice-home".into() },
        GuestMountSpec { tag: "alice-agent-state".into(), guest_path: Some("/agent-state".into()), host_path: "/tmp/demo/alice-agent-state".into() },
    ],
    software: SoftwareProfile { packages: vec!["vim".into()] },
    resources: GuestResources {
        boot_vcpus: 2,
        memory_mib: 512,
        max_vcpus: None,
    },
    storage: GuestStorage {
        overlay_size: OverlaySize::new("2G")?,
    },
    boot: BootArtifacts {
        kernel: "/tmp/vmm-v1.4/libs/vmm/examples/v1.4/artifacts/base/Image".into(),
        initramfs: None,
        firmware: None,
        cmdline: None,
    },
};

let prepared = orchestrator::prepare(PrepareRequest {
    guest: spec,
    namespace,
    network_modes,
})?;
let handle = orchestrator::boot(prepared)?;
handle.ready(&readiness_policy).await?;

// Verify egress networking
let out = orchestrator.exec(&handle, "curl -s -o /dev/null -w '%{http_code}' https://example.com").await?;
assert_eq!(out.exit_code, 0);
assert_eq!(out.stdout.trim(), "200");

// Verify VFS mount wiring
let out = orch.exec(&handle, "readlink ~/.codex").await?;
assert_eq!(out.stdout.trim(), "/agent-state/codex");

// Verify DNS resolution through motlie-vnet
let dns = handle.net_assignment.egress_ipv4.dns;
let out = orch
    .exec(&handle, &format!("nslookup example.com {}", dns))
    .await?;
assert_eq!(out.exit_code, 0);

handle.shutdown().await?;
```

### Usage: interactive SSH ingress (FR-6)

```rust
// The SSH proxy runs as a background task inside the orchestrator.
// Users connect from outside:
//   ssh -p 2222 alice@localhost
//
// The proxy:
//   1. Extracts "alice" as guest identity
//   2. Calls orchestrator.ensure_vm("alice") — boots if needed
//   3. Signs ephemeral cert (Ed25519, principal="alice", 60s TTL)
//   4. Opens russh::client to guest sshd over vsock (cid:2222)
//   5. Bridges PTY, data, window-change bidirectionally
//
// The user lands in a shell inside the guest VM.
```

## Relationship to `examples/v1.3`

`examples/v1.3` is the validated comparison baseline for the VMM-owned harness.

## `v1.4` Refactor Line

`v1.4` is the first line where the validated `v1.3` harness starts to shed
orchestration logic into reusable library modules.

Design intent:

- `v1.3` stays intact as the comparison baseline
- `v1.4` becomes the refactor branch
- reusable logic moves into `libs/vmm/src`
- `repl_host_v1_4` becomes a thin application and test harness

The first `v1.4` target is not a new product capability. It is a structural
change in where the capability lives.

### `v1.4` Requirements

1. Move reusable `repl_host_v1_3` logic into library modules under
   `libs/vmm/src`.
2. Fork `examples/v1.3` into `examples/v1.4`.
3. Refactor `repl_host_v1_4` to consume library services instead of owning the
   orchestration logic directly.
4. Keep `repl_host_v1_3` unchanged for comparison and API-surface review.
5. Use a distinct `v1.4` namespace everywhere so `v1.3` and `v1.4` can run at
   the same time without collisions.

### Expected First Extractions

The first reusable slices expected to move from `repl_host_v1_3` into the
library are:

- typed guest spec / mount / identity modeling
- network allocation and reusable guest identity assignment
- launch artifact generation
- boot / wait-ready / shutdown orchestration
- guestfs provisioning and mount wiring
- validation helpers for automation-first checks

This points to the following likely module layout:

- `libs/vmm/src/spec.rs`
- `libs/vmm/src/network_alloc.rs`
- `libs/vmm/src/artifacts.rs`
- `libs/vmm/src/orchestrator.rs`
- `libs/vmm/src/guestfs.rs`
- `libs/vmm/src/validation.rs`

### `v1.4` Thin-Harness Target

`repl_host_v1_4` should keep only:

- command parsing
- operator-facing output
- example-specific topology choices
- comparison/debug affordances

It should stop owning:

- the boot asset rendering rules
- the reusable launch/shutdown sequencing
- the reusable guest allocation table
- the reusable validation command implementations

### `v1.4` Namespace Rule

`v1.4` must not reuse the `v1.3` runtime namespace.

Expected namespace examples:

- binary:
  - `repl_host_v1_4`
- temp/runtime roots:
  - `/tmp/motlie-vmm-v14-*`
- guest sockets:
  - `/tmp/motlie-vmm-v14-alice.sock`
  - `/tmp/motlie-vmm-v14-alice.vsock`
  - `/tmp/motlie-vmm-v14-alice-api.sock`
- integration/tmux names:
  - `v14-*`

This is required so `v1.3` and `v1.4` can run side by side during the
refactor.

## Allocation Design: CID, IP, and MAC

`libs/vmm` now owns the reviewed guest identity allocator in
[`src/network_alloc.rs`](../src/network_alloc.rs). The contract is no longer
"small `alice`/`bob` demo convenience"; it is a slot-derived identity model
that can support materially more than 7 guests in one harness instance.

### Goal

One stable `slot` must deterministically generate all per-guest identity used
by the composed Motlie-backed flow:

- vsock CID
- admin ingress subnet and host/guest IPv4 pair
- admin MAC
- egress subnet and guest/host/DNS IPv4 layout
- egress MAC
- vhost-user socket path

This keeps the reviewed API small:

- `Ipv4Subnet`
- `Ipv4SubnetPool`
- `GuestNetAllocatorConfig`
- `GuestNetAllocator`
- `GuestNetAssignment`

### Derivation Rule

For one harness process:

- `slot` is the stable per-guest allocation index
- `cid = first_cid + slot`
- `admin_subnet = nth_child(admin_pool.base, admin_pool.guest_prefix_len, slot)`
- `admin_host = admin_subnet + host_offset`
- `admin_guest = admin_subnet + guest_offset`
- `egress_subnet = nth_child(egress_pool.base, egress_pool.guest_prefix_len, slot)`
- `egress_host = egress_subnet + host_offset`
- `egress_guest = egress_subnet + guest_offset`
- `egress_dns = egress_subnet + dns_offset`

MAC identity is also slot-derived:

- admin: `52:54:00:(0xa0 | slot_hi4):slot_mid8:slot_lo8`
- egress: `52:54:00:(0xe0 | slot_hi4):slot_mid8:slot_lo8`

This gives a deterministic per-plane MAC namespace without needing external
state or collision-prone truncation to one byte.

### Capacity Rule

Allocator capacity is computed, not guessed.

Effective capacity is:

- `min(admin_pool.child_capacity, egress_pool.child_capacity, cid_headroom, mac_headroom, max_guests?)`

Where:

- `admin_pool.child_capacity = 2^(guest_prefix_len - base_prefix_len)`
- `egress_pool.child_capacity = 2^(guest_prefix_len - base_prefix_len)`
- `cid_headroom = u32::MAX - first_cid + 1`
- `mac_headroom = 2^20` in the current encoded slot suffix
- `max_guests` is an optional harness/user clamp

The important behavioral property is that exhaustion is explicit and typed.
There is no silent wraparound and no `192.168.255.x` style saturation.

### Default Allocation Shape

The current defaults intentionally separate admin and egress concerns:

- admin pool:
  - base `172.20.0.0/16`
  - per-guest subnet `/30`
  - host offset `1`
  - guest offset `2`
- egress pool:
  - base `10.0.0.0/8`
  - per-guest subnet `/24`
  - host offset `2`
  - guest offset `15`
  - DNS offset `3`

Default capacity under those values:

- admin capacity: `16384` guests
- egress capacity: `65536` guests
- effective default capacity: `16384` guests

This is a deliberate correction to the old `192.168.249.0/24` through
`192.168.255.0/24` stopgap.

### API Rule

The reviewed API must expose both the declarative policy and the resulting
assignment:

```rust
pub struct Ipv4Subnet {
    pub network: Ipv4Addr,
    pub prefix_len: u8,
}

pub struct Ipv4SubnetPool {
    pub base: Ipv4Subnet,
    pub guest_prefix_len: u8,
    pub host_offset: u32,
    pub guest_offset: u32,
    pub dns_offset: Option<u32>,
}

pub struct GuestNetAllocatorConfig {
    pub first_cid: u32,
    pub max_guests: Option<u32>,
    pub socket_dir: PathBuf,
    pub admin_pool: Ipv4SubnetPool,
    pub egress_pool: Ipv4SubnetPool,
}

pub struct GuestNetAssignment {
    pub slot: u32,
    pub cid: u32,
    pub admin_subnet: Ipv4Subnet,
    pub admin_ipv4: AdminIpv4Pair,
    pub admin_mac: [u8; 6],
    pub egress_subnet: Ipv4Subnet,
    pub egress_ipv4: EgressIpv4Layout,
    pub egress_mac: [u8; 6],
    pub vnet_socket_path: PathBuf,
}
```

Required behaviors:

- config validation is explicit
- capacity is queryable before boot
- assignments are stable for the life of one harness run
- exhaustion is machine-readable
- harness UX can expose both config and computed capacity

### Harness UX Rule

The first customer of the public allocation API is `harness_v1_4`.

The harness must expose:

- allocator config via CLI (`--first-cid`, `--max-guests`, `--admin-base`,
  `--admin-guest-prefix`, `--egress-base`, `--egress-guest-prefix`)
- current computed capacity in shell mode
- concrete CID/IP/MAC assignment in `where <guest>`

This keeps capacity planning and debugging in the harness itself rather than
forcing agents or operators to reverse-engineer allocation from code.
Once validated, the extraction sequence should be:

1. move typed config and network allocation into `libs/vmm`
2. move launch artifact rendering and runtime path layout into `libs/vmm`
3. move launch/shutdown/readiness orchestration into `libs/vmm`
4. move guestfs and SSH bridge lifecycle wiring into `libs/vmm`
5. leave example-specific topology and operator REPL UX in `examples/v1.3`
6. add a non-interactive harness mode or binary that uses only library APIs

This keeps the example as a demo/runbook while shrinking its bespoke logic and
simultaneously creates a stable harness surface for future agent-driven Motlie
development.

## Stable Harness Direction

The current `repl_host_v1_3` is good enough for operator debugging, but it is
still too REPL-centric to be the long-term development harness. The stable
harness should have these properties:

- blocking guest bring-up with explicit readiness, not best-effort background
  launch
- a single typed `VmHandle` / `HarnessState` rather than multiple unrelated
  maps in the example
- structured shutdown and validation results rather than stderr-only status
  lines
- deterministic timeouts and error classes for launch, readiness, exec, and
  teardown
- a runnable non-interactive mode suitable for agents, CI, and future tooling

In concrete terms, `repl_host_v1_3` should become a thin interactive client
over the same reusable APIs that power a future automation-first harness.

## Validation Gates Before Extraction

Before code moves from `v1.3` into `libs/vmm`, the `v1.3` flow should prove:

- guest boots reliably with generated launch assets
- admin ingress and egress networking compose correctly
- `motlie-vfs` mounts still work in the composed launch
- guest has outbound internet access through `motlie-vnet`
- startup/shutdown sequencing is deterministic
- Alice/Bob multi-guest flows do not rely on accidental path or ordering hacks
- launch readiness can be expressed as explicit gates rather than "try exec and
  hope the guest is up"
- shutdown returns a structured, machine-usable outcome
- non-interactive command and validation flows do not depend on REPL stderr

If any of those remain unstable, the code should stay in the example until the
behavior settles.

## Resolved Questions

<!-- @claude-vmm 2026-04-04 — Resolved during design session. -->

**Q: Should `libs/vmm` be Cloud Hypervisor-specific in v1, or CH-first with
internal seams?**
A: CH-specific. See NFR-3. No abstraction trait in v1 — no consumers to
justify the complexity. Clean module boundaries are sufficient.

**Q: How much of the current REPL launch flow should remain example-only?**
A: The REPL is a consumer of `libs/vmm`, not part of it. The boundary is
clear: `libs/vmm` owns `prepare` / `boot` / `ready` / `shutdown` / `exec`.
The REPL owns the interactive command loop and operator UX. Examples
become thin wiring over library calls.

**Q: Should cloud-init generation live as a standalone submodule?**
A: Implementation detail of `prepare()`. In v1.2, cloud-init rendering
is tightly coupled to guest config (name, user, mounts, network mode).
Exposing it as a separate public API adds surface with no current
consumer. Keep it in `cloud_init.rs` as an internal module.

**Q: How should guest runtime artifact cleanup be exposed?**
A: Explicit `shutdown()` with deterministic fallback (API → SIGTERM →
SIGKILL), as already proven in v1.2's `shutdown_guest()`. Best-effort
`Drop` on `VmHandle` as a safety net, but callers should not rely on it.

## Open Questions

- Should the SSH proxy listen on a fixed port (2222) or let the caller
  choose? The product doc (`docs/motlie-vmm.md`) uses 2222 as default.
  The library should probably accept a `SocketAddr` in config.
- How should the orchestrator handle concurrent `exec` calls to the same
  guest? Open multiple SSH channels on one connection, or one connection
  per exec? Performance vs complexity tradeoff.
- Should `exec` have a timeout parameter, or should the caller wrap it?
- Should the stable automation harness be a new binary (for example,
  `harness_host`) or a non-interactive mode layered into `repl_host_v1_3`?
- Which readiness gates should be mandatory for `VmHandle::ready(...)` by
  default: API socket, SSH bridge, or full exec-ready?
