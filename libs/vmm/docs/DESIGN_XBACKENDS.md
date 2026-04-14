# motlie-vmm: Cross-Backend Infrastructure Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Add explicit fallback gates for `v1.15` managed guestfs failure, make kernel virtio-driver verification a required `v1.05` exit gate, and formalize the hardened SSH auto-provision checks as `v1.45` acceptance tests |
| 2026-04-14 | @codex-vz | Clarify the Apple Vz execution order as `v1.05` image/build -> `v1.15` guestfs -> `v1.25` egress -> CH-safe refactors -> policy phases -> `v1.45` full VMM, move `v1.05` under `libs/vmm/examples/`, and add measurable exit gates for each vertical slice |
| 2026-04-13 | @codex-vz | Add a first-step Apple Vz image track to the cross-backend sequence: `v1.05` image/build proving first, then `v1.15` guestfs, `v1.25` egress, the cleanup phases, the separate policy phases (`#134`, `#133`), and finally the `v1.45` full `libs/vmm` Vz vertical slice |

## Purpose

`libs/vmm` consumes both `motlie-vfs` and `motlie-vnet` as real infrastructure:

- `examples/v1.4` uses `motlie-vfs` for managed guest filesystem semantics
- `examples/v1.4` uses `motlie-vnet` for guest egress
- the harness validates lifecycle plus those subsystem outcomes together

That means Apple Vz support cannot be designed as a hypervisor-only effort.
`libs/vmm` needs a sequencing document that keeps the stable CH line intact
while the Vz-specific proving work happens underneath it.

## Non-Negotiable Constraints

Every stage in the Vz track must preserve the same host-side product
constraints already required by the CH stack:

- all userspace on the host
- no persistent host configuration changes
- no persistent host traces other than caller-selected backing directories and
  ordinary build/runtime artifacts
- runtime state is ephemeral and tied to host process lifetime

Parity is layered:

- image parity means the Vz guest can boot the same Linux userspace contract
- filesystem parity means the guest still uses managed `motlie-vfs` semantics
- network parity means the guest still uses policy-capable `motlie-vnet`
  semantics
- full backend parity means `prepare/boot/ready/exec/pty/shutdown`,
  provisioning, SSH proxy, guestfs, and egress all line up with `v1.4`

## Execution Order

1. `v1.05` guest image/build PoC in `libs/vmm/examples/v1.05`
2. `v1.15` VFS guestfs PoC in `libs/vfs/examples/v1.15` plus `libs/vfs/vz`
3. `v1.25` VNET egress PoC in `libs/vnet/examples/v1.25` plus `libs/vnet/vz`
4. `motlie-vfs` CH-safe cleanup / transport-boundary refactor
5. `motlie-vnet` CH-safe cleanup / adapter-boundary refactor (`#169`)
6. `#134` VFS policy engine
7. `#133` VNET policy engine
8. `v1.45` full `libs/vmm` Vz vertical slice

This is intentionally evidence-first. `libs/vmm` should not freeze abstractions
for Vz before the image, guestfs, and egress proofs show what is actually
possible.

## Stage Responsibilities

### Stage 0: `v1.05` Guest Image / Build PoC

This stage exists to prove the boot contract required by `vz-runner`.

Responsibilities:

- produce a bootable aarch64 Linux guest for Apple Vz
- prove kernel, initrd, root disk, and NoCloud packaging
- prove cloud-init user and SSH-key injection
- prove that guest-side binaries such as `motlie-vfs-guest` can be present at
  boot

`libs/vmm` implication:

- no `backend::vz` code yet
- only design and artifact-contract consumption

Exit gates:

- guest boots to a reachable login/serial prompt
- cloud-init provisions the intended guest user and SSH key
- the image contains `motlie-vfs-guest`
- the selected kernel is verified to expose the virtio-block, virtio-vsock,
  virtio-net, and, where needed, virtio-fs guest drivers required by later Vz
  slices
- the image path still satisfies the host-impact constraints above

### Stage 1: `v1.15` VFS Guestfs PoC

This stage proves whether Apple Vz can carry the existing managed filesystem
contract.

Responsibilities:

- prove multi-guest / multi-tag guestfs over a Vz-specific transport
- keep `FsServer` in charge of semantics
- preserve overlay visibility and write behavior
- preserve enough op context for `#134`

`libs/vmm` implication:

- still no final backend integration
- learn what readiness and observability signals the future Vz guestfs path
  will need

Exit gates:

- a tagged share becomes visible in the guest
- an overlay write made through the managed path is visible in the guest
- guestfs readiness fires only after the managed path is actually live

Decision gate if this fails:

- if managed guestfs cannot ride a Vz-compatible transport cleanly, Vz does not
  claim `motlie-vfs` parity
- the fallback is an explicitly degraded static `VirtioFS` sharing mode for
  bootstrap/debug use only
- full `v1.45` parity is blocked until a different managed transport path is
  designed and proven

### Stage 2: `v1.25` VNET Egress PoC

This stage proves whether Apple Vz can feed packets through a Rust-owned egress
engine rather than Apple NAT alone.

Responsibilities:

- prove the Vz raw-packet path into the reusable engine
- preserve the no-persistent-host-config property
- preserve enough visibility for `#133`

Expected transport shape:

- `vz-runner` creates a datagram socketpair for
  `VZFileHandleNetworkDeviceAttachment`
- one fd is handed to Apple Vz as the guest NIC attachment
- the peer fd stays in Rust and carries raw L2 Ethernet frames into the
  reusable packet engine
- the PoC validates framing, lifecycle, and observability on that concrete path

`libs/vmm` implication:

- still documentation-only at the VMM layer
- learn what backend-neutral egress signals the future backend will need

Exit gates:

- guest can `curl https://example.com`
- guest DNS activity is visible to the Rust-owned engine
- the packet path remains userspace-only and ephemeral on the host

### Stage 3: CH-Safe Refactors

This stage converts PoC lessons into reusable boundaries without changing
outcomes for the stable CH line.

Responsibilities:

- clean up `motlie-vfs` transport boundaries
- clean up `motlie-vnet` engine/adapter boundaries
- make `libs/vmm` observability and readiness capability-oriented rather than
  CH-transport-oriented

Exit gates:

- `examples/v1.4/repl_host.rs` remains functional on CH
- `examples/v1.4/harness` remains functional on CH
- CH outcomes stay the same even if internal reporting names change

### Stage 4: Policy Phases

This stage restores the product-level semantics that matter beyond raw
transport viability.

Responsibilities:

- `#134` implements filesystem policy and enriched observability on the
  reusable `motlie-vfs` path
- `#133` implements DNS/TCP policy and enriched observability on the reusable
  `motlie-vnet` path

Exit gates:

- policy semantics live in reusable cores, not in backend-specific adapters
- Vz parity claims remain impossible until these semantics are either preserved
  or explicitly scoped down

### Stage 5: `v1.45` Full VMM Slice

This stage integrates the proven filesystem and network stories into the full
`libs/vmm` lifecycle.

Responsibilities:

- `prepare()`, `boot()`, `ready()`, `exec()`, `open_pty()`, and `shutdown()`
- provisioning and SSH proxy compatibility
- guestfs and egress integration using the reviewed lower-layer results

Exit gates:

- the existing auto-provision scenario passes on Vz
- `examples/v1.4/scenarios/auto-provision-ssh.json` passes unchanged on Vz
- `examples/v1.4/integration/repl-auto-provision-smoke.sh` passes unchanged on
  Vz
- the backend exposes parity-capable observability
- the lifecycle works end to end without CH-specific transport assumptions

## Design Consequences For `libs/vmm`

`libs/vmm` should evolve toward capability-style infrastructure reporting:

- guest image ready / not ready
- managed guestfs available / unavailable
- policy-capable egress available / unavailable
- backend-specific transport identity as metadata, not as the product contract

`libs/vmm` should not permanently depend on:

- CH-specific socket existence as the only readiness signal
- raw VirtioFS as equivalent to managed `motlie-vfs`
- Apple NAT as equivalent to policy-capable `motlie-vnet`

## Review Gates

Before merging any cross-backend `libs/vmm` work, verify:

- CH behavior remains stable
- the proposed abstraction matches a proven Vz vertical slice, not only a guess
- host-impact constraints are preserved
- `v1.05`, `v1.15`, and `v1.25` each have explicit measurable evidence
- parity language distinguishes bootstrap/debug capability from real parity
