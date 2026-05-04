# DESIGN_IMAGE_BUILDER

## Purpose

This document records the common guest-image contract that `v1.35` now shares
with CH `v1.3`, and defines the next builder direction after parity.

The canonical post-v1.45 VMM-owned guest image and guest binary plan now lives
in [`../../docs/DESIGN_GUEST_IMAGE.md`](../../docs/DESIGN_GUEST_IMAGE.md).
Use this file as historical v1.35 evidence, not as the owning v1.5 plan.

The goal is not just to remove Tart or reduce duplicated scripting. The goal is
to prevent backend-specific guest semantic drift.

`v1.3` is the current semantic source of truth.

## Current Status

`v1.35` has been realigned to the CH `v1.3` guest contract:

- one generic base image
- per-launch guest identity and runtime mutation
- real in-image `codex` / `claude` installs
- `/agent-state` presented into guest homes via bind mounts
- launch validation based on guest contract behavior, not Vz-specific wrapper or
  cache inventions

This means image convergence is now true at the contract level, even though one
common builder implementation does not exist yet.

## Common Guest Contract

The common image must bake:

- `sshd` with:
  - `TrustedUserCAKeys /etc/ssh/ca/user_ca.pub`
  - `AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u`
- generic `alice` / `bob` guest accounts
- `motlie-agent-state.service`
- `motlie-vmm-vsock-ssh.service`
- `motlie-vfs-guest`
- real CLI installs required by the contract:
  - `@openai/codex`
  - `@anthropic-ai/claude-code`
- validation tooling and guest bootstrap helpers

The common image must not bake:

- per-guest API keys
- per-guest SSH principals files
- per-guest hostnames
- per-guest writable workspace or home state
- backend-specific runtime network state
- launch-time synthetic wrappers to fake bundled software

## Launch-Time Inputs

Launch may inject:

- CA trust material
- per-user principals files
- hostname
- guest-specific `.env`
- writable runtime root
- VFS/VMM mount configuration
- backend-specific runtime network attachment

Launch must not:

- install packages to achieve parity
- fabricate wrapper commands for tools that are supposed to be image-baked
- change the intended guest-visible semantics between CH and Vz

## Agent-State Contract

The guest must see:

- `~/.codex`
- `~/.claude`
- `~/.config/claude-code`

These are presented from `/agent-state` via bind mounts.

The contract is guest-visible and backend-independent:

- tools should experience stable writable directories
- guest tools should not need to know whether CH or Vz is underneath

Implication for builder design:

- the image must always include `motlie-agent-state-setup`
- builder output must not encode backend-specific home-path semantics

## Validation Contract

The common validation contract should assert:

- `/workspace` mount present
- `/agent-state` mount present
- writable:
  - `~/.codex`
  - `~/.claude`
  - `~/.config/claude-code`
- `authorized_keys` present
- DNS works
- outbound internet works
- expected guest `.env` line is present

Validation should not depend on:

- wrapper script contents
- backend-specific cache paths
- runtime package-manager cold start behavior

## Builder Architecture Direction

The next builder should produce one logical guest image and two backend-specific
artifact forms.

Logical stages:

1. common rootfs/image assembly
- package install
- user creation
- service installation
- common guest helper installation
- CLI installation

2. contract verification
- assert the baked image satisfies the common guest contract before backend
  packaging

3. backend emitters
- CH emitter
- Vz emitter

The backend emitters should only adapt artifact format and boot/runtime wiring,
not guest semantics.

### CH Emitter

Produces artifacts equivalent to CH `v1.3` expectations, such as:

- `Image`
- `rootfs.squashfs`
- any CH-specific launch overlay inputs

### Vz Emitter

Produces artifacts equivalent to `v1.35` expectations, such as:

- `disk.img`
- `nvram.bin`
- native source/base VM directory layout

The Vz emitter must stay Tart-free.

## Non-Goals

This design does not require:

- a shared VMM/VFS/VNET runtime abstraction in the same change
- merging CH and Vz launch scripts immediately
- forcing one common host networking backend

Those are separate follow-on refactors.

## Success Criteria

The common image builder is successful when:

1. one guest contract document governs both CH and Vz
2. both backends boot from artifacts emitted from the same common image build
3. guest-visible semantics match for the validated contract
4. no backend-specific wrapper or package-install behavior is needed at launch
5. Tart is not part of the Vz image build path

## Immediate Next Step

Use this document as the contract for:

- keeping `v1.35` aligned with CH `v1.3`
- evaluating future CH changes against a shared guest contract
- designing the eventual common image builder without reintroducing semantic
  drift
