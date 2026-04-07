# v1.4 Library Extraction Harness

`v1.4` starts from the validated `v1.3` harness and changes the goal:

- `v1.3` remains the comparison baseline
- `v1.4` is the refactor line
- reusable orchestration logic moves out of `repl_host.rs` and into
  `libs/vmm/src`
- `repl_host_v1_4` should become a thin application and test harness over
  library services

This branch is intentionally documentation-first. The first `v1.4` checkpoint
defines the extraction plan and namespace rules before adding new prototype
features.

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
- validation helpers for non-interactive harness checks

The resulting shape should look like:

- `libs/vmm/src/spec.rs`
- `libs/vmm/src/network_alloc.rs`
- `libs/vmm/src/artifacts.rs`
- `libs/vmm/src/orchestrator.rs`
- `libs/vmm/src/guestfs.rs`
- `libs/vmm/src/validation.rs`

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

## Near-Term Implementation Order

1. create `examples/v1.4/` by forking `v1.3`
2. rename the `v1.4` runtime namespace
3. extract typed guest/network/artifact helpers into `libs/vmm/src`
4. add a thin `repl_host_v1_4`
5. add `v1.4`-owned smoke coverage

Further prototype features should be recorded in:

- [`libs/vmm/docs/DESIGN.md`](../../docs/DESIGN.md)
- [`libs/vmm/docs/PLAN.md`](../../docs/PLAN.md)
