# motlie-vmm: Cross-Backend Infrastructure Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Initial cross-backend plan for `libs/vmm`: sequence the work as `#170` Vz egress PoC first, `#169` `motlie-vnet` refactor second, `#133` policy engine third, and full `backend::vz` integration last |

## Goal

Make `libs/vmm` ready for multiple hypervisor backends without destabilizing
the current CH + `motlie-vnet` path.

The main constraint is that `motlie-vnet` is already part of real `vmm`
behavior:

- `examples/v1.4` uses it for outbound internet
- the harness validates it
- future backend parity depends on how egress evolves

So the plan must:

- preserve Linux CH behavior first
- gather Vz packet-path evidence before freezing architecture
- keep `#133` visible as the policy/observability layer rather than losing it
  inside a larger refactor

## Issue Order

1. `#170` Vz vertical-slice PoC in `libs/vnet/vz` and `libs/vnet/examples/v1.25`
2. `#169` `motlie-vnet` reusable-core / CH-adapter refactor
3. `#133` policy engine implementation
4. future Vz `libs/vmm` backend vertical slice

## Phase 1: Vz Egress PoC (`#170`)

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

- enough evidence exists to refine or confirm `libs/vnet/docs/DESIGN_XBACKENDS.md`
- we know whether full policy parity is plausible, blocked, or uncertain

## Phase 2: `motlie-vnet` CH-Safe Refactor (`#169`)

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

## Phase 3: Policy Engine (`#133`)

Objective:

- implement DNS/TCP observability and policy control on top of the refactored
  reusable egress engine

`libs/vmm` tasks:

- update parity language so "egress" and "policy-capable egress" are distinct
- identify what future harness coverage should validate for policy-aware
  scenarios

Exit criteria:

- the reusable egress engine owns the `#133` semantics cleanly
- `libs/vmm` can reason about policy-capable egress as an infrastructure
  capability

## Phase 4: Vz Backend Integration

Objective:

- only after egress feasibility and architecture are sufficiently clear, begin
  the full Vz backend slice in `libs/vmm`

Scope:

- backend lifecycle
- provisioning
- SSH proxy / auto-provision
- mount / guest service integration
- egress integration using the agreed `vnet` outcome

Exit criteria:

- Vz is integrated against a real, reviewed egress story rather than an
  unstable or speculative one

## `libs/vmm` Review Checklist

Before merging any cross-backend `vmm` work, verify:

- are we preserving current CH behavior?
- are we depending on capabilities/outcomes rather than CH transport details?
- are we accidentally assuming Apple NAT is sufficient for parity?
- are we keeping `#133` visible in the acceptance story?
- are we avoiding `vmm` example forks until the `vnet` PoC actually justifies
  them?

## Near-Term Recommendation

Start with:

- `#170` first

Then revisit:

- `libs/vnet/docs/DESIGN_XBACKENDS.md`
- `libs/vmm/docs/DESIGN_VZ.md`
- this plan

Only after that should the broader `motlie-vnet` refactor proceed.
