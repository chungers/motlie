# motlie-vfs: Cross-Backend Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Add an explicit `v1.15` failure decision: degraded static `VirtioFS` is allowed only as a bootstrap/debug fallback and does not count as full `motlie-vfs` parity |
| 2026-04-14 | @codex-vz | Rewrite the Vz cross-backend plan into numbered, checkable phases; move the image/build PoC to `libs/vmm/examples/v1.05`; and make the measurable `v1.05` and `v1.15` exit gates explicit |
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image-contract phase before the `v1.15` guestfs PoC, then refactor transport boundaries, then implement the richer policy engine from `#134` |

## Goal

Prove and then generalize a Vz-compatible `motlie-vfs` path without weakening
the existing CH semantics or freezing the wrong abstraction boundary up front.

Reference:

- [`DESIGN_XBACKENDS.md`](./DESIGN_XBACKENDS.md)

## Phase 0: `v1.05` Guest Image / Build PoC

Scope:

- `libs/vmm/examples/v1.05`
- guest artifact build and validation only

Tasks:

- [ ] build a bootable aarch64 guest artifact set for Apple Vz
- [ ] prove NoCloud/cloud-init delivery
- [ ] verify the selected kernel exposes virtio-block, virtio-vsock,
      virtio-net, and, where needed, virtio-fs guest drivers
- [ ] prove `motlie-vfs-guest` is available at boot

Validation:

- [ ] guest boots to a serial or login prompt
- [ ] cloud-init provisions the intended user and SSH key
- [ ] kernel virtio-driver verification is recorded before `v1.05` exits
- [ ] `motlie-vfs-guest` is present in the booted image

## Phase 1: `v1.15` Vz Guestfs PoC

Scope:

- `libs/vfs/examples/v1.15`
- `libs/vfs/vz`

Tasks:

- [ ] prove multi-guest / multi-tag guestfs on Apple Vz
- [ ] keep `FsServer` in the managed I/O path
- [ ] preserve overlay read/write behavior
- [ ] preserve enough request context for future `#134` policy work
- [ ] make an explicit decision if managed guestfs cannot be preserved on Vz:
      block parity or downgrade to static `VirtioFS`

Validation:

- [ ] a tagged share becomes visible in the guest
- [ ] an overlay write becomes visible through the guest mount
- [ ] readiness fires only after the managed path is actually live
- [ ] host-side state remains userspace-only and ephemeral
- [ ] if managed guestfs fails, the docs record that static `VirtioFS` is a
      degraded mode only and does not count as full parity

## Phase 2: CH-Safe `motlie-vfs` Refactor

Tasks:

- [ ] isolate transport-specific code more clearly
- [ ] align guest connector assumptions with lessons from the Vz PoC
- [ ] define backend-neutral readiness signals

Validation:

- [ ] existing `libs/vfs/examples/v1` still behaves correctly
- [ ] existing `libs/vfs/examples/v1.1` still behaves correctly
- [ ] CH-facing `libs/vmm` guestfs behavior remains stable

## Phase 3: `#134` Policy Engine

Tasks:

- [ ] implement enriched filesystem observability and policy in the reusable
      `FsServer` path
- [ ] keep policy semantics out of backend-specific transport adapters

Validation:

- [ ] policy semantics are testable on the reusable server path

## Phase 4: VMM Integration

Tasks:

- [ ] add `libs/vmm/src/guestfs_vz.rs`
- [ ] align backend-neutral guestfs readiness with the proven transport shape

Validation:

- [ ] VMM integration depends on a reviewed standalone Vz guestfs path, not a
      guessed one

## Merge Checklist

- [ ] CH semantics remain intact
- [ ] raw `VirtioFS` is not mislabeled as full `motlie-vfs` parity
- [ ] userspace-only and no-persistent-host-change constraints remain intact
- [ ] `#134` remains visible as a separate semantic phase
