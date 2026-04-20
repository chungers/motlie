# v1.25 Apple Vz Vnet PoC

`v1.25` is the Apple Vz fork of
[`libs/vnet/examples/v1.2`](../v1.2/README.md).

This slice exists to answer two questions quickly:

1. Can the Apple Vz path support the same guest-visible split-network outcome
   that `v1.2` proved on CH?
2. Does Apple Vz expose enough host-side packet visibility and control to
   justify future `motlie-vnet` policy work, or is Apple NAT too opaque?

This is intentionally a PoC:

- backend-specific code is acceptable here
- elegance is not the primary goal
- CH must remain stable while Vz is explored locally in this tree

## Initial Success Criteria

The first `v1.25` milestone is smaller than full CH parity.

It should establish:

1. native Apple Vz boot using the signed runner model proven in
   `libs/vfs/examples/v1.15`
2. clean fresh-clone lifecycle with no persistent host-visible network state
3. guest outbound internet access on Apple Vz
4. a converging guest contract with `v1.2`:
   - same users where possible
   - same uid/gid targets where possible
   - same passwords
   - same packages
   - same agent-state layout
5. evidence about packet-path visibility:
   - what the host can observe/control
   - whether Apple NAT is only bootstrap help or fundamentally insufficient

The important rule is:

- “internet works” is necessary
- it is not sufficient if packet visibility is too weak for future policy work

## Relationship To Other Slices

- `v1.1` is the validated CH multi-guest VFS slice
- `v1.15` is the validated Apple Vz multi-guest VFS slice
- `v1.2` adds CH split-network + `/agent-state`
- `v1.25` is the Apple Vz parallel of `v1.2`

So `v1.25` starts from two already-proven ingredients:

- the `v1.2` guest contract and validation goals
- the `v1.15` native Apple Vz runtime model

## Current Implementation Direction

The first implementation pass in this directory does the following:

- forks the `v1.2` example tree directly into `v1.25`
- imports the native Apple Vz helper / launcher scaffolding from `v1.15`
- keeps the `v1.2` guest-side mount contract:
  - `/home/<user>`
  - `/agent-state`
  - `/workspace`
- keeps the long-term image-convergence goal visible from the start

That means this tree currently contains some inherited CH-oriented files while
the Vz path is being stood up. The intended end state is:

- Vz-native launcher / runtime docs here
- Vz-native validation flow here
- CH-specific launcher details removed from the canonical `v1.25` runbook

## Image Convergence Goal

One of the first outcomes of `v1.25` should be convergence between the CH and
Vz guest images.

Near-term target:

- one logical guest image spec for both platforms
- same guest-visible contract across CH and Vz

Likely still backend-specific at first:

- boot artifact format
- launch-time attachment / seed / provisioning mechanics

So the convergence target is:

- same packages
- same users / uid / gid targets
- same passwords
- same services
- same shell/profile behavior
- same agent-state wiring

not necessarily:

- one byte-identical boot artifact immediately

## Networking Expectations

What CH `v1.2` already proved:

- TAP can keep admin SSH ingress
- `motlie-vnet` can provide a separate egress path
- the host can observe/control that egress backend

What `v1.25` still must prove:

- whether Apple Vz has any usable host-visible packet path
- whether that path is strong enough for future DNS/TCP policy semantics

Current expectation:

- Apple NAT may be enough for bootstrap and basic egress
- Apple NAT alone is probably not enough for the long-term policy model

That is why `v1.25` exists:

- to produce evidence, not just optimism

## Current State

This tree is now scaffolded for Apple Vz work:

- `launch-vz.sh`, `build-vz-runner.sh`, `vz-vsock-runner.m`, and
  `vz.entitlements` are imported from the validated `v1.15` slice as the
  starting native runtime path
- the inherited `v1.2` guest mount YAML and REPL data remain the source of
  truth for the intended guest-visible contract

It is not yet claiming validated end-to-end `v1.2` networking parity on Apple
Vz. That proof is the work of this issue.

## Files

| File | Purpose |
|------|---------|
| `README.md` | Scope, goals, and current Apple Vz status for `v1.25` |
| `IMAGE.md` | Guest-image convergence notes for the Vz fork |
| `VZ-HARNESS.md` | Apple Vz runtime/runbook notes |
| `build-vz-runner.sh` | Build the example-local signed Apple Vz helper |
| `launch-vz.sh` | Native Apple Vz launcher scaffold for the `v1.25` slice |
| `vz-vsock-runner.m` | Current example-local Apple Vz helper |
| `build-guest.sh` | Inherited `v1.2` CH guest-image builder, still to be adapted/converged for Vz |

