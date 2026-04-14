# motlie-vfs: Cross-Backend Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Clarify that `motlie-vfs` parity means the full managed-filesystem semantics, move the Vz image PoC to `libs/vmm/examples/v1.05`, and add measurable gates for `v1.05` and `v1.15` |
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image-contract phase: prove Vz guest image / cloud-init / guest-binary delivery before the `v1.15` guestfs PoC, then keep the cleanup and `#134` policy work sequenced after those proofs |

## Purpose

`motlie-vfs` is already closer to a reusable cross-backend subsystem than
`motlie-vnet`. The main uncertainty is not overlay semantics. It is the
transport and lifecycle path that carries guest filesystem operations between a
Linux guest and the host-side `FsServer`.

This document records the target semantics and the Vz execution order:

1. `v1.05` Vz guest image/build PoC in `libs/vmm/examples/v1.05`
2. `v1.15` Vz guestfs PoC in `libs/vfs/examples/v1.15` plus `libs/vfs/vz`
3. CH-safe transport-boundary cleanup in `motlie-vfs`
4. `#134` policy engine on the clarified core
5. later `libs/vmm` integration

## Non-Negotiable Product Constraints

Any backend that claims `motlie-vfs` parity must preserve:

- all host-side control and service code in userspace
- no persistent host configuration drift
- no persistent host traces other than caller-selected backing directories and
  ordinary build/runtime artifacts
- runtime-only connection state that disappears when the host process exits

## What Parity Means

Parity is not "the guest can see a host directory."

A backend only has `motlie-vfs` parity when it preserves:

- tagged mount routing
- host-side `FsServer` dispatch for guest I/O
- layered memfs plus disk-backed base semantics
- shared named memfs layers spanning multiple mount tags
- whiteouts and synthetic directory behavior
- runtime mutation of overlay state while the guest is live
- event and future policy emission from the same dispatch path

Raw hypervisor-native `VirtioFS` sharing is therefore not parity by itself. It
may still be useful for bootstrap/debug sharing, but it bypasses the managed
filesystem semantics that matter.

## Stage 0: `v1.05` Image / Build PoC

Purpose:

- prove the Vz boot artifact shape before blaming guestfs for image problems

Questions answered here:

- what guest disk artifact boots cleanly under Apple Vz
- how cloud-init is delivered
- how `motlie-vfs-guest` is baked into the guest
- whether the same guest-side systemd/unit contract can come up at boot

Exit gates:

- guest boots to a serial or login prompt
- cloud-init provisions the intended user and SSH key
- `motlie-vfs-guest` is present in the booted image

## Stage 1: `v1.15` Guestfs PoC

Purpose:

- prove whether Apple Vz exposes a host/guest stream path that can carry the
  existing managed guestfs contract end to end

Recommended shape:

- `libs/vfs/examples/v1.15`
- `libs/vfs/vz`
- a minimal `guestfs_vz` wrapper later, only if the standalone PoC succeeds

What must remain true:

- `FsServer` stays in charge of semantics
- the guest still sees tagged mounts and managed overlay behavior
- the path still preserves enough context for `#134`

Exit gates:

- a tagged share is visible in the guest
- an overlay write is visible through the guest mount
- readiness fires only after the managed path is live

## Long-Term Boundary

### Reusable Core

Responsibilities:

- wire operations and results
- dispatch and mount routing
- overlay publication and layer semantics
- event emission and policy evaluation

### Backend-Specific Transport

Responsibilities:

- carry `FsOp` and `FsResult` between guest and host
- own handshake and per-tag connection establishment
- surface readiness and teardown state

Transport must not own:

- overlay semantics
- policy decisions
- mount routing

## Relationship To `#134`

`#134` remains a separate phase because it owns richer policy/event semantics,
not transport feasibility. The Vz PoC only needs to preserve the possibility of
that design by keeping the managed server path intact.

## Relationship To `libs/vmm`

The intended VMM sequence is:

1. prove the Vz image/build path in `libs/vmm/examples/v1.05`
2. prove the standalone `motlie-vfs` Vz path in `libs/vfs/examples/v1.15`
3. learn what transport and readiness boundaries are real
4. then update `libs/vmm` with `guestfs_vz.rs` and backend-neutral readiness

This keeps VMM integration downstream of the filesystem feasibility proof.
