# motlie-vmm: Cross-Backend Infrastructure Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Rewrite the cross-backend plan into numbered, checkable phases with explicit validation gates; move `v1.05` under `libs/vmm/examples/`; and make the Apple Vz execution order image -> VFS -> VNET -> cleanup -> policy -> full `v1.45` explicit |
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image track to the cross-backend plan: prioritize `v1.05` image/build proving before `v1.15` guestfs and `v1.25` egress, then run the cleanup phases and the separate policy phases (`#134`, `#133`) before `v1.45` full Vz integration |

## Goal

Execute the Apple Vz support track without destabilizing the stable CH `v1.4`
line and without committing `libs/vmm` to speculative abstractions before the
lower-level Vz proofs exist.

References:

- [`DESIGN_XBACKENDS.md`](./DESIGN_XBACKENDS.md)
- [`DESIGN_VZ.md`](./DESIGN_VZ.md)
- [`DESIGN_GUEST_IMAGE.md`](./DESIGN_GUEST_IMAGE.md)
- `libs/vfs/docs/DESIGN_XBACKENDS.md`
- `libs/vnet/docs/DESIGN_XBACKENDS.md`

## Phase 0: `v1.05` Guest Image / Build PoC

Scope:

- `libs/vmm/examples/v1.05`
- guest artifact build scripts and notes only

Tasks:

- [ ] build a bootable aarch64 guest artifact set for Apple Vz
- [ ] prove the kernel, initrd, raw root disk, and NoCloud disk contract
- [ ] prove cloud-init user creation and SSH-key injection
- [ ] prove `motlie-vfs-guest` is present in the guest image

Validation:

- [ ] guest boots to a serial or login prompt
- [ ] cloud-init provisions the intended user and SSH key
- [ ] `motlie-vfs-guest` is present in the booted image
- [ ] host-side state remains userspace-only and ephemeral

References:

- DESIGN_XBACKENDS Stage 0
- DESIGN_GUEST_IMAGE.md

## Phase 1: `v1.15` VFS Guestfs PoC

Scope:

- `libs/vfs/examples/v1.15`
- `libs/vfs/vz`

Tasks:

- [ ] prove multi-guest / multi-tag Vz guestfs transport viability
- [ ] keep `FsServer` in the managed I/O path
- [ ] preserve overlay write/read semantics
- [ ] capture the readiness signal shape needed by future `libs/vmm`

Validation:

- [ ] a tagged share becomes visible in the guest
- [ ] an overlay write through the managed path becomes visible in the guest
- [ ] readiness fires only after the managed path is actually live
- [ ] host-side state remains userspace-only and ephemeral

References:

- DESIGN_XBACKENDS Stage 1
- `libs/vfs/docs/DESIGN_XBACKENDS.md`

## Phase 2: `v1.25` VNET Egress PoC

Scope:

- `libs/vnet/examples/v1.25`
- `libs/vnet/vz`

Tasks:

- [ ] prove the Vz raw-packet path into a Rust-owned engine
- [ ] capture the egress observability shape needed by future `libs/vmm`
- [ ] confirm the path still satisfies the no-persistent-host-config rule

Validation:

- [ ] guest can `curl https://example.com`
- [ ] guest DNS activity is visible to the Rust-owned engine
- [ ] host-side state remains userspace-only and ephemeral

References:

- DESIGN_XBACKENDS Stage 2
- `libs/vnet/docs/DESIGN_XBACKENDS.md`

## Phase 3: `motlie-vfs` CH-Safe Refactor

Scope:

- transport-boundary cleanup in `motlie-vfs`
- `libs/vmm` guestfs readiness/observability cleanup

Tasks:

- [ ] remove CH-only guestfs assumptions from `libs/vmm` readiness language
- [ ] preserve current CH guestfs outcomes in `examples/v1.4`
- [ ] align the future VMM boundary with the Vz guestfs findings

Validation:

- [ ] `examples/v1.4/repl_host.rs` still works on CH
- [ ] `examples/v1.4/harness` still works on CH

References:

- DESIGN_XBACKENDS Stage 3
- `libs/vfs/docs/PLAN_XBACKENDS.md`

## Phase 4: `motlie-vnet` CH-Safe Refactor (`#169`)

Scope:

- engine/adapter split in `motlie-vnet`
- `libs/vmm` egress observability cleanup

Tasks:

- [ ] remove CH-only egress assumptions from `libs/vmm`
- [ ] preserve current CH egress outcomes in `examples/v1.4`
- [ ] align the future VMM boundary with the Vz egress findings

Validation:

- [ ] `examples/v1.4/repl_host.rs` still works on CH
- [ ] `examples/v1.4/harness` still works on CH
- [ ] egress validation still passes end to end on CH

References:

- DESIGN_XBACKENDS Stage 3
- `libs/vnet/docs/DESIGN_XBACKENDS.md`

## Phase 5: `#134` VFS Policy Engine

Tasks:

- [ ] preserve policy/event semantics in the reusable guestfs core
- [ ] update VMM parity language so "managed guestfs" and
      "policy-capable guestfs" are distinct capabilities

Validation:

- [ ] policy semantics are testable without backend-specific glue

References:

- DESIGN_XBACKENDS Stage 4

## Phase 6: `#133` VNET Policy Engine

Tasks:

- [ ] preserve DNS/TCP policy semantics in the reusable egress core
- [ ] update VMM parity language so "guest has egress" and
      "policy-capable egress" are distinct capabilities

Validation:

- [ ] policy semantics are testable without backend-specific glue

References:

- DESIGN_XBACKENDS Stage 4

## Phase 7: `v1.45` Full VMM Slice

Scope:

- full `backend::vz` lifecycle and integration work

Tasks:

- [ ] implement `prepare/boot/ready/exec/pty/shutdown`
- [ ] integrate provisioning and SSH proxy
- [ ] integrate guestfs and egress using the reviewed lower-layer outcomes

Validation:

- [ ] `auto-provision-ssh.json` passes on Vz
- [ ] lifecycle and observability are backend-neutral at the VMM contract layer

References:

- DESIGN_XBACKENDS Stage 5
- DESIGN_VZ.md

## Merge Checklist

- [ ] CH remains stable
- [ ] each phase is justified by concrete evidence from the prior one
- [ ] bootstrap/debug paths are not mislabeled as full parity
- [ ] host-impact constraints remain intact
