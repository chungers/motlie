# motlie-vmm: Cross-Backend Infrastructure Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Initial cross-backend design note for `libs/vmm`: treat `motlie-vnet` as core VMM infrastructure, sequence the work as Vz egress PoC first (`#170`), then `motlie-vnet` CH-safe refactor (`#169`), then the policy engine (`#133`), then full Vz backend integration |

## Purpose

`libs/vmm` now depends on `motlie-vnet` as core infrastructure, not as an
optional sidecar:

- `examples/v1.4` uses `motlie-vnet` for outbound guest internet
- harness validation checks depend on egress correctness
- the future Apple Vz backend needs an answer for policy-capable egress, not
  just "some networking"

That means cross-backend planning cannot live only in `libs/vnet`.

This document explains how `libs/vmm` should consume the evolving `vnet`
architecture and why the execution order is:

1. `#170` Vz vertical-slice egress PoC first
2. `#169` `motlie-vnet` reusable-core / CH-adapter refactor second
3. `#133` policy engine implementation on top of that refactor
4. full `backend::vz` vertical slice in `libs/vmm` after the egress path is
   proven

## Why `#170` Comes First

The key uncertainty is not whether `libslirp` is portable. It is whether Apple
Vz can expose a packet path that lets Rust preserve the same egress policy and
observability semantics that `motlie-vnet` is being designed to own.

If we refactor `motlie-vnet` first without that evidence, we risk:

- introducing adapter boundaries based on the wrong Vz assumptions
- over-generalizing public API too early
- taking regression risk in the Linux CH path without first proving there is a
  real second backend worth accommodating

So `libs/vmm` should explicitly treat the Vz egress PoC as a design gate.

## `motlie-vnet` As VMM Infrastructure

From the `libs/vmm` point of view, `motlie-vnet` is responsible for:

- outbound guest internet
- eventually DNS/TCP observability and policy control from `#133`
- preserving the product constraint:
  - no persistent host network configuration changes
  - all userspace
  - runtime state is ephemeral during host lifetime

That means `libs/vmm` should depend on egress capabilities, not on CH transport
details.

Required `libs/vmm` abstractions over time:

- guest has egress / guest lacks egress
- egress is policy-capable / not policy-capable
- backend-neutral egress observability
- backend-neutral validation criteria in the harness

`libs/vmm` should not permanently depend on assumptions like:

- `runtime_paths.vnet_socket` always exists
- outbound internet implies a CH/vhost-user socket
- "Motlie vnet" is the only valid label for egress

## VMM Implications Of The New Issue Order

### Stage 1: `#170` Vz Egress PoC

No `libs/vmm` integration is required yet.

Expected output from `#170` that matters to `libs/vmm`:

- is policy-capable egress on Vz feasible at all?
- what packet/lifecycle/observability signals are available?
- what does the eventual Vz egress adapter need from `vmm`?
- can the no-host-config-drift requirement still hold?

At this stage, `libs/vmm` remains CH-first and stable.

### Stage 2: `#169` `motlie-vnet` Refactor

This is where `libs/vmm` should begin preparing for backend-neutrality, but
still without a live Vz backend.

Expected `libs/vmm` work after or alongside `#169`:

- remove CH-specific egress assumptions from observability and harness code
- define capability-style reporting:
  - egress available
  - policy-capable egress
  - backend-specific transport label
- keep `examples/v1.4` validation unchanged in outcome, not necessarily in
  transport labels

### Stage 3: `#133` Policy Engine

Once `motlie-vnet` has the right reusable core shape, `#133` becomes a direct
VMM concern because `libs/vmm` acceptance now includes policy-capable egress as
part of backend parity.

Implications for `libs/vmm`:

- harness coverage eventually needs policy-aware scenarios
- backend parity should mean not just internet access, but preserved policy
  semantics where the backend claims full parity

### Stage 4: Full Vz Backend Slice

Only after the egress path is proven should `libs/vmm` absorb Vz into the full
lifecycle stack:

- `prepare()`
- `boot()`
- `ready()`
- `exec()`
- `open_pty()`
- `shutdown()`
- provisioning
- SSH proxy / auto-provision
- egress

## Acceptance Implications For `libs/vmm`

When Vz becomes real in `libs/vmm`, parity should be judged in layers:

### Layer A: Lifecycle

- backend boots
- readiness works
- exec/PTy/shutdown work

### Layer B: Guest Services

- guest image / cloud-init / ssh / users / mount wiring still behave correctly

### Layer C: Egress

- outbound internet works
- host impact remains ephemeral and userspace-only

### Layer D: Policy Parity

- only claim full parity if the final egress path preserves the `#133`
  DNS/TCP policy and observability semantics

This layered acceptance model matters because `#170` may prove:

- lifecycle feasibility is high
- basic internet egress is feasible
- full policy parity is blocked or delayed

`libs/vmm` should record those layers honestly instead of collapsing them into a
binary "Vz works / Vz does not work."

## Relationship To Existing Docs

This document depends on and should stay aligned with:

- `libs/vnet/docs/DESIGN.md`
- `libs/vnet/docs/DESIGN_XBACKENDS.md`
- `libs/vmm/docs/DESIGN_VZ.md`

`DESIGN_VZ.md` explains the Vz backend direction.

This document explains the surrounding `vmm` infrastructure sequencing so the
backend does not get implemented against unstable or incorrect assumptions about
how `motlie-vnet` will evolve.
