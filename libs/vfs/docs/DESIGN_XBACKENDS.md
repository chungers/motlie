# motlie-vfs: Cross-Backend Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image-contract phase: prove Vz guest image / cloud-init / guest-binary delivery in `examples/v1.05` before the `v1.15` guestfs PoC, then keep the cleanup and `#134` policy work sequenced after those proofs |
| 2026-04-13 | @codex-vz | Initial cross-backend design note for `libs/vfs`: preserve the managed `FsServer` / overlay / policy semantics across CH and Apple Vz, sequence Vz guestfs PoC work before cleanup, and keep `#134` as the policy-engine follow-up rather than collapsing it into transport work |

## Purpose

`motlie-vfs` is already closer to a reusable cross-backend subsystem than
`motlie-vnet`.

Reusable today:

- `FsServer` owns the semantic core
- overlay and shared-layer behavior are host-side userspace state
- policy belongs at the dispatch layer
- the guest-side contract is still "Linux FUSE over a request/response
  transport"

The main backend-specific risk is therefore not filesystem semantics. It is the
transport and lifecycle path used to connect a Linux guest running under a
specific hypervisor to the host-side `FsServer`.

This document records the long-term target and the implementation order:

1. Vz image/build vertical-slice PoC first (`v1.05`)
2. Vz guestfs vertical-slice PoC second (`v1.15`)
3. `motlie-vfs` transport cleanup and backend boundary refactor third
4. `#134` policy engine implementation on top of the clarified core
5. full `libs/vmm` Vz backend integration after the filesystem path is proven

## Non-Negotiable Product Constraints

Any backend path that claims parity for `motlie-vfs` must preserve:

- all host-side control and service code in userspace
- no persistent host configuration drift or kernel/network setup changes
- no persistent host traces other than caller-selected backing directories and
  normal build/runtime artifacts
- runtime-only connection state that disappears when the host process exits

For `vfs`, these constraints are stricter than simple host-directory sharing.
The backend must preserve the same managed semantics as the CH path where
parity is claimed.

## What "Parity" Means For `motlie-vfs`

Parity is not satisfied by "the guest can see a host directory."

A backend only has `motlie-vfs` parity when it can preserve:

- tagged mount routing
- host-side `FsServer` dispatch for guest I/O
- layered memfs + disk-backed base semantics
- shared named memfs layers spanning multiple mount tags
- whiteouts and synthetic directory behavior
- runtime mutation of overlay state while the guest is live
- guest-visible semantics through ordinary filesystem operations
- policy and event emission from the same dispatch path

That is why raw hypervisor-native `VirtioFS` sharing is not parity by itself.
It may be useful as a bootstrap aid, but it bypasses the managed `FsServer`
path and therefore loses the key overlay and policy semantics.

## Reusable Core Versus Backend-Specific Path

### Likely Reusable Without Major Semantic Change

- `core/server.rs`
- `core/overlay.rs`
- wire types in `core/op.rs`
- event emission in `core/event.rs`
- the future policy engine from `#134`

These are already shaped like reusable filesystem semantics rather than
Cloud-Hypervisor-only glue.

### Likely Backend-Specific Or Needing Refactor

- `vsock/`
- the guest-side connector shape in `client/guest.rs`
- host-side VMM service wrapping in `libs/vmm/src/guestfs.rs`
- readiness assumptions based on the current CH-oriented socket/handshake path

The delivery model is likely still reusable:

- guest FUSE client
- request/response transport
- host-side listener/handler

The current `vsock` realization is not.

## Why The Vz Image PoC Comes First

The very first unknown is not even the guestfs transport. It is whether the
current Linux guest contract can be repackaged for Apple Vz without dragging CH
boot assumptions along with it.

Before asking whether guestfs semantics survive, we need to know:

- what guest disk artifact boots cleanly under Vz
- how cloud-init is delivered on Vz
- how `motlie-vfs-guest` is baked into or otherwise delivered into the guest
- whether the same guest-side systemd/unit contract can come up at boot

That is why the first step should be a Vz-specific image/build PoC forked from
`examples/v1`.

## Why The Vz Guestfs PoC Comes Second

The biggest unknown is whether Apple Vz provides a clean enough host/guest
stream path to carry the existing `FsOp` / `FsResult` traffic while preserving:

- guest mount viability
- readiness/lifecycle control
- overlay semantics
- enough op context for the future `#134` policy design

If we refactor first without that evidence, we risk freezing the wrong adapter
boundary around CH-specific assumptions or around an imagined Vz path that does
not match reality.

So the first step should be a dirty vertical slice that is allowed to be
backend-specific and non-final.

## Vz Vertical Slice Shape

The recommended proving-ground sequence is:

1. `libs/vfs/examples/v1.05`
   - forked from `examples/v1`
   - narrow Vz image/build proving ground
   - proves guest boot artifact shape, NoCloud delivery, and guest-binary
     installation strategy
2. `libs/vfs/examples/v1.15`
   - forked from `examples/v1.1`
   - multi-guest / multi-tag Vz guestfs proof
3. `libs/vfs/vz`
   - Vz-specific transport / bridge experiment
   - no claim of final architecture
4. `libs/vmm/src/guestfs_vz.rs`
   - minimal lifecycle wrapper only after the standalone `vfs` PoC shows the
     path is viable

The `v1.05` image/build PoC should answer:

- can the guest-side Linux contract from the CH line be reused at all?
- do we reuse the same guest userspace payload but different boot packaging?
- how does Vz consume cloud-init compared with the CH launch-time overlay
  seeding path?
- should guest binaries remain baked into the image rather than fetched at
  first boot?

The `v1.15` guestfs PoC should answer:

- can a Linux guest on Vz mount through the same `FuseClient` model?
- can host-side `FsServer` stay in charge of filesystem semantics?
- can overlay reads/writes and shared layers behave the same way?
- what transport abstraction, if any, is actually needed?
- does the request path still surface enough context for `#134`?

## Proposed Long-Term Boundaries

### Core

Responsibilities:

- define wire operations and results
- own dispatch and mount routing
- own overlay publication and layer semantics
- own event emission and policy evaluation

Core must not know:

- Cloud Hypervisor vhost-vsock details
- Apple Vz helper-process details
- any specific host listener path shape

### Transport Adapter

Responsibilities:

- carry `FsOp` / `FsResult` between guest and host
- own handshake and per-mount connection establishment
- expose enough lifecycle signals for readiness and teardown

Transport adapters should be thin. They should not own:

- overlay semantics
- policy decisions
- mount routing
- synthetic directory logic

### Guest Mounter

Responsibilities:

- create guest mount points
- instantiate a `FuseClient`
- connect per-tag transport sessions

The guest mounter should evolve toward transport-pluggable connection setup, but
that should happen after the Vz PoC teaches the real requirements.

## Relationship To `#134`

`#134` is the policy-engine follow-up, not part of the initial Vz transport
proof.

That issue should remain visible and separate because it owns:

- richer pre-op and post-op policy evaluation
- caller identity propagation
- enriched event context
- deny thresholds and chained policy composition

The Vz vertical slice only needs to preserve the possibility of that design.
It does not need to implement `#134` fully before feasibility is known.

## Relationship To `libs/vmm`

`libs/vmm` currently wraps guestfs through a CH-shaped path. That is acceptable
today, but it should not define the future boundary.

The intended sequence is:

1. prove a standalone Vz image/build path in `examples/v1.05`
2. prove a standalone `vfs` Vz guestfs path in `examples/v1.15`
3. learn what transport and readiness boundaries are real
4. then update `libs/vmm`:
   - `guestfs_vz.rs`
   - backend-neutral readiness language
   - eventual `backend::vz` integration

This keeps VMM integration downstream of the filesystem feasibility proof
instead of turning `libs/vmm` into the first experimental surface.
