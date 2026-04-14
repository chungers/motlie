# motlie-vmm: Cross-Backend Infrastructure Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Expand the cross-backend plan to include `motlie-vfs`: prioritize `v1.15` guestfs PoC first, `v1.25` egress PoC second, then the `motlie-vfs` / `motlie-vnet` cleanup phases and the separate policy phases (`#134`, `#133`) before `v1.45` full Vz integration |
| 2026-04-13 | @codex-vz | Initial cross-backend plan for `libs/vmm`: sequence the work as `#170` Vz egress PoC first, `#169` `motlie-vnet` refactor second, `#133` policy engine third, and full `backend::vz` integration last |

## Goal

Make `libs/vmm` ready for multiple hypervisor backends without destabilizing
the current CH + `motlie-vfs` + `motlie-vnet` path.

The main constraint is that both subsystems are already part of real `vmm`
behavior:

- `examples/v1.4` uses `motlie-vfs` for guest mounts and runtime layering
- `examples/v1.4` uses `motlie-vnet` for outbound internet
- the harness validates both
- future backend parity depends on how filesystem and egress evolve

So the plan must:

- preserve Linux CH behavior first
- gather Vz guestfs and packet-path evidence before freezing architecture
- keep `#134` and `#133` visible as separate semantic phases rather than losing
  them inside larger refactors

## Issue Order

1. Vz guestfs PoC in `libs/vfs/vz` and `libs/vfs/examples/v1.15`
2. `#170` Vz egress PoC in `libs/vnet/vz` and `libs/vnet/examples/v1.25`
3. `motlie-vfs` cross-backend cleanup / adapter refactor
4. `#169` `motlie-vnet` reusable-core / CH-adapter refactor
5. `#134` VFS policy engine
6. `#133` VNET policy engine
7. future Vz `libs/vmm` backend vertical slice in `examples/v1.45`

## Phase 1: Vz Guestfs PoC (`v1.15`)

Objective:

- determine whether Apple Vz can expose a guestfs path that preserves the
  long-term `motlie-vfs` managed semantics

Expected outputs:

- concrete notes on what Vz guestfs transport APIs can and cannot do
- evidence on whether `FsServer` semantics survive unchanged
- evidence on whether userspace / no-persistent-host-change constraints hold
- a judgment on whether `#134` semantics remain plausible on Vz

`libs/vmm` work in this phase:

- none required in code
- consume findings into documentation only

Exit criteria:

- enough evidence exists to refine or confirm
  `libs/vfs/docs/DESIGN_XBACKENDS.md`
- we know whether managed guestfs parity is plausible, blocked, or uncertain

## Phase 2: Vz Egress PoC (`#170`)

Objective:

- determine whether Apple Vz can expose a packet path that preserves the
  long-term `motlie-vnet` policy direction

Expected outputs:

- concrete notes on what Vz networking APIs can and cannot do
- evidence on whether guest packet TX/RX can flow through a Rust-owned path
- evidence on whether no-host-config-drift still holds
- a judgment on whether `#133` semantics are plausible on Vz

`libs/vmm` work in this phase:

- none required in code
- consume findings into documentation only

Exit criteria:

- enough evidence exists to refine or confirm
  `libs/vnet/docs/DESIGN_XBACKENDS.md`
- we know whether full policy parity is plausible, blocked, or uncertain

## Phase 3: `motlie-vfs` CH-Safe Refactor

Objective:

- split `motlie-vfs` into reusable semantic core + cleaner transport boundary
  without changing Linux behavior

`libs/vmm` tasks:

- identify CH-specific guestfs assumptions in readiness / observability
- define the backend-neutral guestfs signals `vmm` will eventually need
- preserve existing validation outcomes for `examples/v1.4`

Required validation:

- existing CH guestfs behavior inside `examples/v1.4`
- existing `libs/vfs/examples/v1` and `v1.1` behavior where still applicable
- eventual `v1.15` lessons

Exit criteria:

- CH remains stable
- `vmm` no longer bakes unnecessary CH guestfs assumptions into future
  cross-backend interfaces

## Phase 4: `motlie-vnet` CH-Safe Refactor (`#169`)

Objective:

- split `motlie-vnet` into reusable egress engine + CH transport adapter without
  changing Linux behavior

`libs/vmm` tasks:

- identify CH-specific egress assumptions in harness / observability
- define the backend-neutral egress signals `vmm` will eventually need
- preserve existing validation outcomes for `examples/v1.4`

Required validation:

- `libs/vmm/examples/v1.4/repl_host.rs`
- `libs/vmm/examples/v1.4/harness`
- relevant `v1.4` integration smoke scripts that exercise egress

Exit criteria:

- CH remains stable
- `vmm` no longer bakes unnecessary CH transport assumptions into future
  cross-backend interfaces

## Phase 5: VFS Policy Engine (`#134`)

Objective:

- implement filesystem observability and policy control on top of the
  refactored reusable guestfs path

`libs/vmm` tasks:

- update parity language so "mounts work" and "policy-capable guestfs" are
  distinct
- identify what future harness coverage should validate for policy-aware
  filesystem scenarios

Exit criteria:

- the reusable guestfs core owns the `#134` semantics cleanly
- `libs/vmm` can reason about policy-capable guestfs as an infrastructure
  capability

## Phase 6: VNET Policy Engine (`#133`)

Objective:

- implement DNS/TCP observability and policy control on top of the refactored
  reusable egress engine

`libs/vmm` tasks:

- update parity language so "egress" and "policy-capable egress" are distinct
- identify what future harness coverage should validate for policy-aware
  networking scenarios

Exit criteria:

- the reusable egress engine owns the `#133` semantics cleanly
- `libs/vmm` can reason about policy-capable egress as an infrastructure
  capability

## Phase 7: Vz Backend Integration

Objective:

- only after guestfs and egress feasibility and architecture are sufficiently
  clear, begin the full Vz backend slice in `libs/vmm`

Scope:

- backend lifecycle
- provisioning
- SSH proxy / auto-provision
- mount / guest service integration
- egress integration using the agreed `vnet` outcome

Exit criteria:

- Vz is integrated against real, reviewed filesystem and egress stories rather
  than unstable or speculative ones

## `libs/vmm` Review Checklist

Before merging any cross-backend `vmm` work, verify:

- are we preserving current CH behavior?
- are we preserving current CH guestfs semantics?
- are we depending on capabilities/outcomes rather than CH transport details?
- are we keeping `#134` visible in the acceptance story?
- are we accidentally assuming Apple NAT is sufficient for parity?
- are we keeping `#133` visible in the acceptance story?
- are we sequencing `v1.15` -> `v1.25` -> `v1.45` rather than jumping directly
  to `vmm` example forks?

## Near-Term Recommendation

Start with:

- `v1.15` first
- then `#170`

Then revisit:

- `libs/vfs/docs/DESIGN_XBACKENDS.md`
- `libs/vnet/docs/DESIGN_XBACKENDS.md`
- `libs/vmm/docs/DESIGN_VZ.md`
- this plan

Only after that should the broader `motlie-vfs` / `motlie-vnet` cleanup
proceed.
