# motlie-vfs: Cross-Backend Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image-contract phase: execute a `v1.05` image/build PoC before the `v1.15` guestfs PoC, then refactor transport boundaries, then implement the richer policy engine from `#134` |
| 2026-04-13 | @codex-vz | Initial cross-backend plan for `libs/vfs`: execute a Vz guestfs vertical slice first (`v1.15` + `libs/vfs/vz`), then refactor transport boundaries, then implement the richer policy engine from `#134` |

## Goal

Make `motlie-vfs` usable across CH and Apple Vz without weakening the existing
CH semantics or prematurely freezing the wrong abstraction boundary.

This plan intentionally separates three concerns:

1. feasibility of a Vz image/build path
2. feasibility of a Vz guestfs path
3. cleanup of `motlie-vfs` transport boundaries
4. richer policy and observability from `#134`

## Issue Order

1. Vz image/build vertical-slice PoC in `libs/vfs/examples/v1.05`
2. Vz guestfs vertical-slice PoC in `libs/vfs/vz` and
   `libs/vfs/examples/v1.15`
3. `motlie-vfs` cross-backend transport cleanup and CH-safe refactor
4. `#134` VFS policy engine implementation
5. full `libs/vmm` guestfs / `backend::vz` integration

## Phase 1: Vz Image / Build PoC

Objective:

- determine whether the current Linux guest contract can be repackaged for
  Apple Vz without keeping Cloud Hypervisor-specific boot assumptions

Scope:

- `libs/vfs/examples/v1.05`
- Vz-specific image/build helpers only

Required success criteria:

- a Vz-suitable guest artifact boots
- NoCloud/cloud-init delivery is proven in a Vz-compatible form
- `motlie-vfs-guest` can be baked into or otherwise delivered into the guest in
  a repeatable way
- the guest-side service/unit contract can come up at boot
- runtime state remains userspace-only and ephemeral

Non-goals:

- no guestfs transport proof yet
- no final image pipeline yet
- no full `libs/vmm` API yet

Exit criteria:

- feasibility of the guest image / boot contract is judged with concrete
  evidence
- we know what can be reused from the CH guest payload and what must differ in
  Vz packaging

## Phase 2: Vz Guestfs PoC

Objective:

- determine whether a Linux guest on Apple Vz can use the existing managed
  `FsServer` semantics through a Vz-specific transport path

Scope:

- `libs/vfs/examples/v1.15`
- `libs/vfs/vz`
- enough guest image / launch glue to prove end-to-end mount behavior

Required success criteria:

- tagged mounts connect and become ready
- guest can read and write through the managed server path
- overlay visibility works
- runtime state is userspace-only and ephemeral
- no persistent host config changes are required
- the request path still exposes enough operation context for future `#134`
  policy work

Non-goals:

- final transport trait
- final `libs/vmm` API
- full multi-guest orchestration
- full `#134` implementation

Exit criteria:

- feasibility is judged as plausible, blocked, or uncertain with concrete
  evidence
- the PoC teaches what should move into a transport boundary

## Phase 3: `motlie-vfs` Cleanup

Objective:

- refactor transport-specific code so the core remains semantic and
  backend-neutral while preserving CH behavior

Likely work:

- isolate the current `vsock` adapter more clearly
- loosen `client/guest.rs` connector assumptions
- define backend-neutral readiness signals for guestfs connections
- keep CH examples and validation behavior stable

Required validation:

- existing `libs/vfs/examples/v1`
- existing `libs/vfs/examples/v1.1`
- Vz image/build learnings from `v1.05`
- Vz PoC learnings from `v1.15`
- current `libs/vmm` guestfs harness checks that depend on the same path

Exit criteria:

- CH remains stable
- the reusable core is clearer
- the adapter boundary is informed by the PoC rather than guessed up front

## Phase 4: `#134` Policy Engine

Objective:

- implement enriched filesystem observability and access control in the
  reusable `FsServer` path

Scope:

- pre-op and post-op policy pipeline
- caller identity propagation
- enriched events and deny thresholds
- policy composition

Constraint:

- policy must live in the reusable server/dispatch path, not in a backend
  transport adapter

Exit criteria:

- CH uses the new policy path through the reusable core
- the design remains compatible with the Vz path proven in earlier phases

## Phase 5: VMM Integration

Objective:

- integrate the proven filesystem path into `libs/vmm`

Likely work:

- `libs/vmm/src/guestfs_vz.rs`
- backend-neutral guestfs readiness wording
- eventual `backend::vz` lifecycle integration

Constraint:

- do not start this phase until the standalone `vfs` Vz path has real evidence

## Review Checklist

Before merging cross-backend `vfs` work, verify:

- are we preserving current CH semantics?
- are we keeping overlay and policy logic in the reusable core?
- are we avoiding raw `VirtioFS` pass-through as a false parity claim?
- are we preserving the userspace / no-persistent-host-change constraints?
- are we keeping `#134` visible as a separate semantic phase?
