# motlie-vmm: Cross-Backend Infrastructure Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image track to the cross-backend sequence: `v1.05` image/build proving first, then `v1.15` guestfs, `v1.25` egress, the cleanup phases, the separate policy phases (`#134`, `#133`), and finally the `v1.45` full `libs/vmm` Vz vertical slice |
| 2026-04-13 | @codex-vz | Expand the cross-backend sequencing to include `motlie-vfs` as core VMM infrastructure: prioritize a `v1.15` Vz guestfs PoC first, then a `v1.25` Vz egress PoC, then the `motlie-vfs` / `motlie-vnet` cleanup phases and the separate policy phases (`#134`, `#133`) before the eventual `v1.45` full `libs/vmm` Vz vertical slice |
| 2026-04-13 | @codex-vz | Initial cross-backend design note for `libs/vmm`: treat `motlie-vnet` as core VMM infrastructure, sequence the work as Vz egress PoC first (`#170`), then `motlie-vnet` CH-safe refactor (`#169`), then the policy engine (`#133`), then full Vz backend integration |

## Purpose

`libs/vmm` now depends on both `motlie-vfs` and `motlie-vnet` as core
infrastructure, not as optional sidecars:

- `examples/v1.4` uses `motlie-vfs` for guest filesystem mounts and runtime
  layering
- `examples/v1.4` uses `motlie-vnet` for outbound guest internet
- harness validation depends on both
- the future Apple Vz backend needs answers for both:
  - managed guestfs semantics, not just raw host sharing
  - policy-capable egress, not just "some networking"

That means cross-backend planning cannot live only in `libs/vnet` or only in
`libs/vfs`.

This document explains how `libs/vmm` should consume the evolving `vfs` and
`vnet` architecture and why the execution order is:

1. Vz image/build PoC first (`v1.05`)
2. Vz guestfs PoC second (`v1.15`)
3. Vz egress PoC third (`v1.25`)
4. `motlie-vfs` cleanup / adapter refactor
5. `motlie-vnet` cleanup / adapter refactor (`#169`)
6. `#134` VFS policy engine
7. `#133` VNET policy engine
8. full `backend::vz` vertical slice in `libs/vmm` / `examples/v1.45`

## Why `v1.05` Comes First

The first uncertainty on Apple Vz is even lower level than guestfs transport:
whether the current Linux guest contract can be repackaged for Vz at all.

Before asking whether managed guestfs survives, we need a concrete answer for:

- guest disk artifact shape under Vz
- cloud-init / NoCloud delivery under Vz
- guest-binary baking or injection strategy
- guest boot-time service viability

This is why the sequence should start with a Vz-specific image/build proof.

## Why `v1.15` Comes Second

The first uncertainty on Apple Vz is not the `motlie-vfs` semantic core. It is
whether a Linux guest on Vz can preserve the same managed guestfs behavior
through a Vz-specific host/guest transport path.

If we refactor first without that evidence, we risk:

- introducing adapter boundaries based on guessed Vz assumptions
- over-generalizing public API too early
- taking regression risk in the Linux CH path without first proving there is a
  real second backend path worth accommodating

This is why the `motlie-vfs` vertical slice should come before the networking
slice. It follows the same shape as the CH lineage:

- `v1` / `v1.1`: filesystem-first proving ground
- `v1.2`: networking expansion
- `v1.4`: full VMM integration

For Apple Vz, the parallel line should be:

- `v1.05`: image/build vertical slice
- `v1.15`: VFS guestfs vertical slice
- `v1.25`: VNET egress vertical slice
- `v1.45`: full VMM vertical slice

## `motlie-vfs` As VMM Infrastructure

From the `libs/vmm` point of view, `motlie-vfs` is responsible for:

- guest mount routing and readiness
- overlay and shared-layer semantics
- future filesystem policy and enriched observability from `#134`
- preserving the product constraints:
  - all userspace on the host side
  - no persistent host configuration changes
  - no persistent host traces other than selected backing directories
  - runtime state that is ephemeral during host lifetime

That means `libs/vmm` should depend on guestfs capabilities, not on the current
CH-shaped socket path.

Required `libs/vmm` abstractions over time:

- guestfs available / guestfs unavailable
- managed guestfs semantics / raw host sharing only
- policy-capable guestfs / not policy-capable
- backend-neutral guestfs observability
- backend-neutral validation criteria in the harness

`libs/vmm` should not permanently depend on assumptions like:

- the current Unix socket listener shape is universal
- current CH-style mount readiness is the only valid signal
- raw VirtioFS pass-through is equivalent to `motlie-vfs`

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

### Stage 1: `v1.05` Vz Image / Build PoC

Minimal or no `libs/vmm` integration is required yet.

Expected output that matters to `libs/vmm`:

- can the CH guest userspace contract be reused under Vz?
- what image artifacts must Vz boot from?
- how should NoCloud / cloud-init be delivered?
- how should guest binaries and services be installed or baked in?

At this stage, `libs/vmm` remains CH-first and stable.

### Stage 2: `v1.15` Vz Guestfs PoC

Minimal or no `libs/vmm` integration is required yet.

Expected output that matters to `libs/vmm`:

- is managed guestfs on Vz feasible at all?
- what host/guest transport and readiness signals are available?
- can the current `FsServer` semantics survive unchanged?
- what does the eventual `guestfs_vz.rs` adapter need from `vmm`?

At this stage, `libs/vmm` remains CH-first and stable.

### Stage 3: `v1.25` Vz Egress PoC

Again, `libs/vmm` should prefer documentation-level consumption of the PoC
findings over early code changes.

Expected output:

- is policy-capable egress on Vz feasible at all?
- what packet/lifecycle/observability signals are available?
- what does the eventual Vz egress adapter need from `vmm`?
- can the no-host-config-drift requirement still hold?

### Stage 4: `motlie-vfs` Cleanup

This is where `libs/vmm` should begin preparing for backend-neutral guestfs
readiness and observability, but still without a live Vz backend.

Expected `libs/vmm` work after or alongside the cleanup:

- remove CH-specific guestfs assumptions from readiness language
- define capability-style reporting for guestfs connection state
- preserve existing `examples/v1.4` outcomes for CH

### Stage 5: `#169` `motlie-vnet` Refactor

This is where `libs/vmm` should begin preparing for backend-neutral egress, but
still without a live Vz backend.

Expected `libs/vmm` work after or alongside `#169`:

- remove CH-specific egress assumptions from observability and harness code
- define capability-style reporting:
  - egress available
  - policy-capable egress
  - backend-specific transport label
- keep `examples/v1.4` validation unchanged in outcome, not necessarily in
  transport labels

### Stage 6: `#134` Policy Engine

Once `motlie-vfs` has the right transport boundary, `#134` becomes a direct VMM
concern because backend parity now includes filesystem policy and observability.

Implications for `libs/vmm`:

- harness coverage eventually needs policy-aware filesystem scenarios
- backend parity should mean not just "mounts work", but preserved policy
  semantics where the backend claims full parity

### Stage 7: `#133` Policy Engine

Once `motlie-vnet` has the right reusable core shape, `#133` becomes a direct
VMM concern because `libs/vmm` acceptance now includes policy-capable egress as
part of backend parity.

Implications for `libs/vmm`:

- harness coverage eventually needs policy-aware networking scenarios
- backend parity should mean not just internet access, but preserved policy
  semantics where the backend claims full parity

### Stage 8: Full Vz Backend Slice

Only after the filesystem and egress paths are proven should `libs/vmm` absorb
Vz into the full lifecycle stack:

- `prepare()`
- `boot()`
- `ready()`
- `exec()`
- `open_pty()`
- `shutdown()`
- provisioning
- SSH proxy / auto-provision
- guestfs
- egress

## Acceptance Implications For `libs/vmm`

When Vz becomes real in `libs/vmm`, parity should be judged in layers:

### Layer A: Lifecycle

- backend boots
- readiness works
- exec/PTY/shutdown work

### Layer B: Guest Services

- guest image / cloud-init / ssh / users / mount wiring still behave correctly
- managed guestfs semantics remain intact

### Layer C: Egress

- outbound internet works
- host impact remains ephemeral and userspace-only

### Layer D: Policy Parity

- only claim full parity if the filesystem path preserves the `#134`
  filesystem policy and observability semantics
- only claim full parity if the final egress path preserves the `#133`
  DNS/TCP policy and observability semantics

This layered acceptance model matters because the Apple Vz slices may prove:

- lifecycle feasibility is high
- guestfs feasibility is high
- basic internet egress is feasible
- full policy parity is blocked or delayed

`libs/vmm` should record those layers honestly instead of collapsing them into a
binary "Vz works / Vz does not work."

## Relationship To Existing Docs

This document depends on and should stay aligned with:

- `libs/vfs/docs/DESIGN.md`
- `libs/vfs/docs/DESIGN_XBACKENDS.md`
- `libs/vnet/docs/DESIGN.md`
- `libs/vnet/docs/DESIGN_XBACKENDS.md`
- `libs/vmm/docs/DESIGN_VZ.md`

`DESIGN_VZ.md` explains the Vz backend direction.

This document explains the surrounding `vmm` infrastructure sequencing so the
backend does not get implemented against unstable or incorrect assumptions about
how `motlie-vfs` and `motlie-vnet` will evolve.
