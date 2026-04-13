# v1.4 Library Extraction Harness

`v1.4` starts from the validated `v1.3` harness and changes the goal:

- `v1.3` remains the comparison baseline
- `v1.4` is the refactor line
- reusable orchestration logic moves out of `repl_host.rs` and into
  `libs/vmm/src`
- `repl_host_v1_4` should become a thin application and test harness over
  library services

`v1.4` is no longer documentation-only. The current slice now includes:

- a real `repl_host_v1_4` example entrypoint
- `v1.4`-namespaced `build-guest.sh` and `launch-ch.sh`
- extracted guestfs provisioning in `libs/vmm/src/guestfs.rs`
- a real rootless automation binary at `examples/v1.4/harness/main.rs`
- a thin harness flow that calls library `prepare()`, `boot()`,
  `VmHandle::ready(...)`, `VmHandle::exec(...)`, and `VmHandle::shutdown()`
- a first PTY/session slice through `VmHandle::open_pty(...)` and
  `harness_v1_4 pty`

Repeatable validation entrypoints now are:

- run these commands from the repo root, for example:
  - `cd /tmp/vmm-v1.4`

- harness lifecycle smoke:
  - `./target/debug/examples/harness_v1_4`
- harness PTY smoke:
  - `./target/debug/examples/harness_v1_4 pty`
- harness machine-readable result output:
  - `./target/debug/examples/harness_v1_4 --result-json /path/to/result.json`
- harness interactive/manual mode:
  - `./target/debug/examples/harness_v1_4 shell`
- harness shell smoke:
  - `./libs/vmm/examples/v1.4/integration/harness-shell-smoke.sh`
  - drives `harness_v1_4 shell` with a saved command script
  - boots `alice` and `bob`
  - runs `validate` for both
  - then verifies proxy SSH, writable mounted state, and outbound fetches for
    both guests
- harness isolation smoke:
  - `./libs/vmm/examples/v1.4/integration/harness-isolation-smoke.sh`
  - runs two separate harness shells concurrently
  - boots `alice` in both
  - proves instance-specific namespace, socket, and proxy isolation

## Recorded PTY Session

A checked-in recorded validation session now lives under:

- [`assets/pty-agent-validation.svg`](./assets/pty-agent-validation.svg)
- [`assets/pty-agent-validation.cast`](./assets/pty-agent-validation.cast)

These artifacts come from the saved `pty-agent-validation` run that:

- waits for package-manager quiescence
- shows one clean `sudo -n apt-get update`
- launches `codex`
- captures the Codex welcome screen

GitHub-friendly snapshot:

![pty-agent-validation](./assets/pty-agent-validation.svg)

The checked-in SVG is a Rust-native export from the harness-rendered terminal
snapshot. The checked-in `.cast` remains the replay/source artifact.

Detailed repeatable runbooks live in:

- [`HARNESS.md`](./HARNESS.md)

The direction from here is:

- the harness becomes the primary driver
- scripted scenarios and future interactive/manual operation should both sit on
  top of the same `libs/vmm` lifecycle and PTY APIs
- `repl_host_v1_4` remains useful during the transition, but should not gain
  unique control-plane logic
- `harness_v1_4 shell` is now the preferred ad-hoc/manual entrypoint
- [`setup-multiguest.harness`](./setup-multiguest.harness) is the saved
  command script for bringing up the standard two-guest `v1.4` flow under the
  harness shell
- `RuntimeNamespace` in `libs/vmm` now owns:
  - default root resolution through `MOTLIE_VMM_ROOT` or the platform temp dir
  - per-process namespace generation
  - guest vsock service socket naming

Operational note:

- `harness_v1_4` now allocates a per-run namespace/demo root/proxy port and
  prints them at startup, so repeated or concurrent harness runs do not collide
  on fixed filenames
- both `harness_v1_4` and `repl_host_v1_4` accept `--root <dir>` and honor
  `MOTLIE_VMM_ROOT`, so they no longer depend on a hardcoded `/tmp` root
- `repl_host_v1_4` now supports `where [guest]` to print the current runtime
  roots and per-guest artifact paths
- `harness_v1_4 --result-json ...` now writes a structured result artifact for
  the `smoke` scenario, including named checks and a library-owned
  observability snapshot

The intended end state is two parallel proof points:

1. a full Motlie-backed path that validates guest behavior end to end through
   the `v1.4` harness and scripts
2. a minimal “hello world” path showing that the same library API can start a
   simple standard Cloud Hypervisor guest without Motlie guestfs or Motlie
   userspace vnet backing

It is still intentionally incomplete compared with `v1.3`:

- `repl_host_v1_4` is still narrower than the old `v1.3` operator surface
- backend shutdown is now library-owned and TERM-first, but cleaner CH-native
  shutdown/reporting remains a quality follow-up

## Scope

`v1.4` must:

1. extract reusable logic from `examples/v1.3/repl_host.rs` into `libs/vmm`
2. fork the `v1.3` example line into `examples/v1.4`
3. refactor `repl_host_v1_4` to consume library services instead of owning the
   orchestration logic directly
4. keep `v1.3` as-is for comparison, regression checking, and API-surface
   review
5. use a distinct `v1.4` namespace everywhere so `v1.3` and `v1.4` can run
   side by side without collisions

## Extraction Target

The reusable logic expected to move into `libs/vmm/src` first is:

- guest spec and runtime-path modeling
- guest network allocation and reusable identity assignment
- launch artifact generation:
  - cloud-init
  - mounts.yaml
  - launch script / launch args
- boot / wait-ready / shutdown orchestration
- guestfs provisioning and mount wiring
- on-demand guest auto-provisioning from incoming SSH principals
- validation helpers for non-interactive harness checks
- guest reporting helpers that combine Cloud Hypervisor host-side counters with
  guest-side health probes

The resulting shape should look like:

- `libs/vmm/src/spec.rs`
- `libs/vmm/src/network_alloc.rs`
- `libs/vmm/src/artifacts.rs`
- `libs/vmm/src/orchestrator.rs`
- `libs/vmm/src/guestfs.rs`
- `libs/vmm/src/provisioning.rs`
- `libs/vmm/src/validation.rs`
- `libs/vmm/src/reporting.rs`

`repl_host_v1_4` should then become a thin layer that:

- parses operator commands
- calls library services
- prints human-readable status
- keeps example-specific demo topology only

## v1.3 Comparison Contract

`v1.3` is intentionally preserved for:

- comparing `repl_host` size and complexity before/after extraction
- reviewing the new library API surface against a known-good harness
- regression checking when `v1.4` behavior diverges
- side-by-side smoke testing during the refactor

Do not rewrite `v1.3` to use the new library modules as part of `v1.4`.

## Namespace Rules

All `v1.4` bins, scripts, runtime paths, and helper names must move to a
`v1.4`-specific namespace so `v1.3` and `v1.4` can run simultaneously.

Examples:

- binaries:
  - `repl_host_v1_4`
- guest image labeling:
  - `v1.4` in MOTD / docs / builder output
- temp/runtime roots:
  - `/tmp/motlie-vmm-v14-*`
- socket names:
  - `/tmp/motlie-vmm-v14-alice.sock`
  - `/tmp/motlie-vmm-v14-alice.vsock`
  - `/tmp/motlie-vmm-v14-alice-api.sock`
- tmux / integration harness names:
  - `v14-*`

The `v1.4` namespace change is part of the design, not cleanup polish. It is
required to support side-by-side comparison with `v1.3`.

## Success Criteria

The first successful `v1.4` slice should show:

1. `examples/v1.4/` exists as a fork of `v1.3`
2. at least one reusable orchestration slice is moved into `libs/vmm/src`
3. `repl_host_v1_4` is smaller than `repl_host_v1_3` because it calls library
   helpers instead of inlining orchestration logic
4. `v1.3` still runs unchanged
5. `v1.3` and `v1.4` can be launched side by side without path or socket
   collisions

Current status against those criteria:

- `examples/v1.4/` exists and is namespaced
- Phase 1 and 2 reusable slices are in `libs/vmm/src`
- `repl_host_v1_4` is a materially thinner operator shell than `v1.3`
- `v1.3` remains unchanged
- `examples/v1.4/harness/main.rs` now boots a guest rootlessly, waits for
  guestfs and SSH bridge readiness through the library lifecycle API, validates
  VFS mounts, validates outbound HTTPS over rootless `vhost-user` egress, and
  shuts the guest down

## Automatic Provisioning From SSH Principals

`v1.4` now supports on-demand guest provisioning driven by incoming SSH
principals through the library-owned `libs/vmm/src/provisioning.rs` registry.
In `repl_host_v1_4`, this mode is disabled by default and must be enabled
explicitly with `auto-provision on`.

Example flow:

- `ssh alice@localhost` -> use existing `alice` guest if already known
- `ssh bob@localhost` -> provision and boot a new `bob` guest if missing
- `ssh joe@localhost` -> allocate a new guest identity, boot it, and continue
  the SSH session once the bridge is ready, when auto-provisioning is enabled
- `ssh joe@localhost` again -> reuse the same provisioned guest and stable slot
  assignment within the current harness run
- `boot alice` -> still boots a guest explicitly regardless of whether
  auto-provisioning is currently on or off

Current implementation notes:

- each guest name gets one stable `GuestNetAssignment`
- assignments are retained for the lifetime of the harness process
- assignments are reused across guest shutdown/reboot within that process
- new guest names consume the next free slot
- the SSH proxy can now call into the provisioner before opening a guest
  session, so external localhost SSH and harness `proxy_exec` both exercise the
  same resolve-or-create path when auto-provisioning is enabled
- `examples/v1.4/scenarios/auto-provision-ssh.json` is the regression scenario
  covering first-contact provisioning and guest reuse
- `integration/repl-auto-provision-smoke.sh` is the external-SSH regression for
  the real `ssh -p <proxy_port> joe@localhost uname -s` flow through
  `repl_host_v1_4`, and it also verifies that manual `boot` continues to work
  with auto-provisioning both off and on

## Host Lifetime Policy

`repl_host_v1_4` and the interactive harness shell now treat operator
disconnects as detach, not shutdown.

This is intentional and should carry forward into the real vmm host:

- stdin EOF must not stop the SSH proxy
- `SIGHUP` must not stop the SSH proxy
- existing guests must stay running after the controlling terminal goes away
- new SSH logins must still work after the operator disconnects
- only explicit `quit` / `exit` or process termination signals such as
  `SIGINT` / `SIGTERM` should trigger guest shutdown

This policy belongs in the host application layer rather than the core
`libs/vmm` library. The library should provide proxy/provisioning primitives;
the host decides whether losing its REPL client means `detach` or `terminate`.

Current hardening notes:

- delayed auto-provision after detach exposed a separate AF_UNIX path-budget
  constraint: compact namespace prefixes and socket directories are required so
  first-contact guest boots still fit `sun_path` under long explicit roots
- `repl_host_v1_4` now supervises the proxy task and restarts it if the task
  exits unexpectedly
- `examples/v1.4/harness/shell.rs` applies the same policy so the manual
  harness shell does not tear down active guests on stdin close
- `integration/repl-headless-hardening-smoke.sh` is the regression that closes
  REPL stdin deliberately, then verifies real external SSH auto-provisioning and
  reconnects still work

The assignment should include:

- guest slot
- vsock CID
- admin ingress subnet/IP pair when admin ingress is enabled
- admin MAC
- egress MAC
- vhost-user socket path
- runtime namespace roots

The allocator must fail clearly once the configured capacity is exhausted rather
than silently saturating into collisions.

## Future Reporting Phase

After the first extraction slices, `v1.4` should add guest reporting and
metrics collection so the harness can answer questions like:

- is the VM up and healthy?
- how much host-visible memory is assigned?
- what device counters are moving?
- is guest CPU, disk, and network activity progressing during a test?

The intended model is:

- use Cloud Hypervisor host-side reporting first for VMM-visible state:
  - `--api-socket`
  - `--event-monitor`
  - `/api/v1/vm.info`
  - `/api/v1/vm.counters`
- use guest-side probes over the existing SSH exec path for guest-visible
  metrics such as:
  - CPU utilization
  - guest-used memory
  - filesystem usage
  - process-level health

This split is important: Cloud Hypervisor can report VMM/device state and
counters, but not full guest OS semantics by itself.

## Near-Term Implementation Order

1. create `examples/v1.4/` by forking `v1.3`
2. rename the `v1.4` runtime namespace
3. extract typed guest/network/artifact helpers into `libs/vmm/src`
4. add a thin `repl_host_v1_4`
5. add a programmatic `v1.4` harness bootstrap
6. add automatic guest provisioning from SSH principal
7. add `v1.4`-owned smoke coverage
8. add host-side and guest-side reporting/metrics collection

Current status against that order:

- steps 1-4 are now in progress in code
- `repl_host_v1_4` is intentionally minimal and currently boots guests through
  `ChShellBackend`
- `examples/v1.4/harness/main.rs` is now the first real non-REPL `v1.4`
  automation substrate

Further prototype features should be recorded in:

- [`libs/vmm/docs/DESIGN.md`](../../docs/DESIGN.md)
- [`libs/vmm/docs/PLAN.md`](../../docs/PLAN.md)
- [`libs/vmm/docs/API.md`](../../docs/API.md)

## Future Union-Binary Prototype

`v1.4` also reserves a dedicated phase for a special distribution mode where
the harness binary embeds an opinionated guest image in its ELF `.rodata`
section and boots from memfd-backed artifacts.

Important placement:

- image construction becomes stable enough after Phase 2, once launch artifacts
  and runtime layout are library-owned
- the prototype becomes practical after the programmatic harness bootstrap,
  because the non-REPL harness can validate boot/exec/network/shutdown in a
  repeatable way

So this feature belongs after the harness bootstrap phase, not before it.
