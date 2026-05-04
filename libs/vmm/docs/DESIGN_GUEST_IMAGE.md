# motlie-vmm Guest Image And Guest Binary Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-04 | @codex-vz | Record the v1.5 host-runtime convergence boundary: CH and VZ keep external VM runners while VFS and egress/VNET services are embedded VMM runtime backends |
| 2026-05-03 | @codex-vz | Define guest functionality conformance as VFS memfs views, apt-backed egress, and Codex/Claude startup in the shared v1.5 harness |
| 2026-05-03 | @codex-vz | Add the Linux CH artifact emitter for the common v1.5 guest contract and make CH egress setup explicit v1.5 platform adaptation |
| 2026-05-03 | @codex-vz | Consolidate v1.5 manual/interactive operation into `harness_v1_5`, deprecate the v1.5 REPL entrypoint, and require SSH auto-provision toggling to live in the harness shell |
| 2026-05-02 | @codex-vz | Establish `libs/vmm/examples/v1.5` as the unified CH/VZ guest-image line, define VMM ownership of guest image and guest binary assembly, and record the canonical v1.5 guest binary placement |

## Purpose

This document is the VMM-owned source of truth for the post-v1.45 guest image
and guest binary convergence work.

`libs/vmm/examples/v1.5` must demonstrate the same capabilities as CH `v1.4`
and VZ `v1.45`, but through:

- one common guest contract
- one common guest filesystem/rootfs payload where the target architecture
  permits it
- one common seed schema
- one common guest service graph
- one common harness and scenario set

Backend-specific code belongs at the emitter, host launcher, host runner, or
host network adapter layer. It must not leak into guest-visible semantics unless
the platform forces that difference and the docs name why.

## Ownership Boundary

`motlie-vmm` owns the product guest image.

That includes:

- `libs/vmm/examples/v1.5`
- image/rootfs assembly scripts
- backend emitters for CH and VZ artifacts
- cloud-init and seed schema
- guest systemd units and boot sequencing
- guest binary packaging and installation paths
- harness scenarios that certify the guest contract end to end
- the single interactive harness shell used for manual validation
- VMM-owned guest runtime source under `libs/vmm/src/guest`
- VMM-owned v1.5 guest binary entrypoints under `libs/vmm/bins/v1.5`

`motlie-vfs` remains the filesystem library.

That includes:

- reusable filesystem core
- guest mount orchestration APIs such as `client::guest`
- public API bug fixes or functionality-gap fixes required by the VMM-owned
  v1.5 guest runtime

`motlie-vnet` remains the network library.

That includes:

- reusable egress engine and backend adapters
- CH vhost-user integration
- VZ userspace egress integration
- future L7 proxy work, including MITM-TLS and self-signed certificate support
- public API bug fixes or functionality-gap fixes required by the VMM-owned
  v1.5 guest runtime

VMM composes those libraries into a bootable guest product. VFS and VNET should
not own the v1.5 guest image, seed schema, or harness directory.

Do not add `libs/vfs/bins/v1.5`, `libs/vfs/examples/v1.5`,
`libs/vnet/bins/v1.5`, or `libs/vnet/examples/v1.5` for this convergence work.
If v1.5 exposes a VFS or VNET gap, fix the reusable library API in place and
keep the guest binary, image builder, boot scripts, and harness under VMM.

## Canonical v1.5 Guest Binary Strategy

The historical binaries are already outside the example trees:

```text
libs/vfs/bins/v1.1/motlie-vfs-guest.rs
libs/vfs/bins/v1.15/motlie-vfs-guest.rs
```

They are Cargo binaries named:

```text
motlie-vfs-guest-v1_1
motlie-vfs-guest-v1_15
```

Those names describe historical example lineage. v1.5 must not fork those files
into `libs/vmm/examples/v1.5/bin` as another long-lived copy, and it must not
create a new `libs/vfs/bins/v1.5` lineage.

The v1.5 path is immediate VMM ownership:

1. Add the canonical guest mounter source under
   `libs/vmm/bins/v1.5/motlie-vfs-guest.rs`.
2. Put reusable guest runtime code under `libs/vmm/src/guest/` as the entrypoint
   grows beyond a thin wrapper.
3. Wire the Cargo binary from `libs/vmm/Cargo.toml` as
   `motlie-vfs-guest-v1_5`. The name describes the guest-side mount role, not
   ownership by the VFS crate.
4. Build that binary for the guest target during v1.5 image assembly.
5. Install it into the guest image as:
   `/opt/motlie/v1.5/guest/bin/motlie-vfs-guest`.
6. Expose the stable runtime path:
   `/usr/local/bin/motlie-vfs-guest`.
7. Keep `v1_1` and `v1_15` available only as historical compatibility
   binaries until v1.5 proves parity.
8. Treat any required VFS or VNET changes as reusable library fixes, not as a
   new VFS/VNET-owned v1.5 binary or example slice.

The canonical v1.5 binary should select behavior from runtime configuration,
not from its historical filename.

Default runtime inputs:

```text
/etc/motlie/v1.5/backend.env
/etc/motlie-vfs/mounts.yaml
```

Example `backend.env`:

```text
MOTLIE_BACKEND=ch
MOTLIE_VFS_TRANSPORT=vsock
MOTLIE_VFS_HOST_CID=2
MOTLIE_VFS_PORT=5000
MOTLIE_NET_BACKEND=ch-vhost-user
MOTLIE_SSH_VSOCK_PORT=2222
```

or:

```text
MOTLIE_BACKEND=vz
MOTLIE_VFS_TRANSPORT=vsock
MOTLIE_VFS_HOST_CID=2
MOTLIE_VFS_PORT=5000
MOTLIE_NET_BACKEND=vz-userspace
MOTLIE_SSH_VSOCK_PORT=2222
```

If a hard platform limitation requires separate guest binaries temporarily,
install them with explicit names under `/opt/motlie/v1.5/guest/bin/` and keep
`/usr/local/bin/motlie-vfs-guest` as the stable selector. Do not overload one
path with different backend-specific contents.

## v1.5 Guest Filesystem Layout

The guest image should use stable VMM-owned paths:

```text
/opt/motlie/v1.5/guest/bin/
  motlie-vfs-guest
  motlie-agent-state-setup
  motlie-vmm-vsock-ssh-loop
  motlie-control-plane-ready

/etc/motlie/v1.5/
  backend.env

/etc/motlie-vfs/
  mounts.yaml

/usr/local/bin/
  motlie-vfs-guest
  motlie-agent-state-setup
  motlie-vmm-vsock-ssh-loop
  motlie-control-plane-ready
```

`/usr/local/bin` is the compatibility surface used by systemd units and
operators. `/opt/motlie/v1.5` is the versioned product payload.

## Common Image Build Process

The v1.5 builder should be a matrix, not one host assumption:

```text
1. Build guest Linux binaries.
2. Assemble common rootfs.
3. Verify the guest contract before backend packaging.
4. Emit CH artifacts.
5. Emit VZ artifacts.
6. Build and sign host-side backend artifacts.
7. Run the CH/VZ harness matrix.
```

Guest Linux binaries are built for the guest architecture. The primary v1.5
validation target should be arm64 first because Apple VZ on Apple Silicon
requires an arm64 guest.

Host macOS artifacts such as `vz-vsock-runner` are host binaries. They are
built and signed by a macOS job or developer machine and are not installed into
the Linux guest image.

Expected host artifact boundary:

```text
artifacts/host/darwin-arm64/
  vz-vsock-runner

artifacts/host/linux-arm64/
  motlie-vnet-vhost-user
```

`vz-vsock-runner` remains a host macOS runner artifact because it is the current
Objective-C/Virtualization.framework boundary. VZ userspace egress is not a
guest payload and is no longer a required launched helper in the v1.5 harness
path. It is a VMM host-runtime backend backed by `motlie_vnet::slirp`.
There is no standalone `vz_egress_helper_v1_5` artifact. The only transitional
subprocess form is `harness_v1_5 vz-egress`, used by the VZ image-builder path
while customizing a base VM.

## Host Runtime Boundary And CH/VZ Parallels

The v1.5 convergence target is one VMM host runtime shape with backend-specific
runner adapters clearly isolated:

| Concern | CH Track | VZ Track | Convergence Boundary |
|---------|----------|----------|----------------------|
| VM runner | External `cloud-hypervisor` process via `launch-ch.sh` | External `vz-vsock-runner` process via `launch-vz.sh` | External runners are acceptable until both can be represented by typed Rust backends |
| VFS host service | Embedded `FilesystemBacking::MotlieVfs` | Embedded `FilesystemBacking::MotlieVfs` | Same VFS server, same guest mounter, backend-specific transport only |
| Egress host service | Embedded `NetworkBacking::MotlieVnet` on Linux | Embedded `NetworkBacking::VzUserspaceEgress` on macOS | Same VMM runtime owns service lifecycle; platform-specific virtio transport differs |
| Guest egress setup | CH service configures the vhost-user/default-route NIC | VZ obtains DHCP/default route from the VZ file-handle NIC/libslirp path | No VNET guest binary; guest-visible network contract stays backend-neutral |
| SSH control plane | VMM SSH proxy over guest bridge | VMM SSH proxy over VZ userspace TCP host-forward | First-contact waits only for `interactive-ready` |

This boundary is intentional. CH and VZ may have different process adapters for
the hypervisor runner, but long-lived host services such as VFS and egress must
be owned by the VMM runtime so lifecycle, observability, shutdown, and future
multi-backend convergence can be reasoned about in one place.

## v1.5 Script Layout And Rust Extraction Seams

All convergence scripts for both backend paths live under
`libs/vmm/examples/v1.5`:

```text
common-contract.sh
  Shared constants and shell helpers. This is the temporary shell form of the
  future Rust image/launch contract.

build-guest.sh
  Current VZ-native image packaging path for the common guest payload. It builds
  the VMM-owned guest mounter once with `--no-default-features --features
  guest-vfs`, installs it, verifies the marker, and emits contract metadata.

build-image.sh
  Common v1.5 builder entrypoint. It currently dispatches to the VZ and CH
  shell emitters and is the CLI seam that should become the Rust image builder.

build-ch-artifacts.sh
  Linux-only CH artifact emitter for the same common guest payload. It builds
  the VMM-owned guest mounter once, assembles the Debian squashfs rootfs, emits
  the CH kernel/rootfs/guest-contract files under `artifacts/base`, and installs
  explicitly marked CH egress setup files for the vhost-user NIC path.

launch-vz.sh
  VZ-specific host adaptation: Apple Virtualization.framework runner, native
  disk clone, consumption of the VMM-owned userspace egress socket, and
  SSH-based dynamic seed refresh. The default v1.5 harness path must not spawn
  a separate egress process. The VMM/harness runtime owns VZ egress lifecycle.

launch-ch.sh
  CH-specific host adaptation: Cloud Hypervisor kernel/rootfs wiring, ext4
  runtime overlay, CH network devices, and NoCloud seed placement.

harness/
  Single v1.5 validation and operator entrypoint. `harness_v1_5 shell` replaces
  the historical REPL for manual boot, external SSH validation, PTY checks,
  status inspection, and SSH auto-provision toggling. The harness owns the
  backend selector (`--backend vz|ch`); backend differences must flow through
  that selector instead of forking the harness.
```

Backend-specific launchers may differ only in host artifact wiring and dynamic
seed transport. Guest-visible paths, service names, contract markers, and
`backend.env` schema come from the common contract. When the Rust builder and
launcher are introduced, `common-contract.sh` should become typed structures and
the two launch scripts should collapse into backend emitters over those
structures.

Do not add a v1.5 `repl_host` replacement. Any missing interactive capability
belongs in the harness shell so the same control surface can drive VZ locally
and CH on a separate host. Shell auto-provisioning must remain explicit state:
default off for manual sessions unless requested, toggleable with
`auto-provision on|off|status`, and documented separately from scenario-owned
auto-provision validation.

Guest functionality conformance is the saved multiguest harness scenario, not
only first-contact readiness. The exit suite must include:

- VFS/FUSE memfs layer views for `/home/<user>`, `/workspace`, and
  `/agent-state`, including guest write visibility.
- Multi-guest workspace isolation.
- Backend internet egress through DNS/HTTPS and `sudo -n apt-get update`.
- Codex and Claude CLI startup with no OS-level execution errors.

Those checks are common harness semantics. If CH and VZ need different host
plumbing to satisfy them, keep that difference in backend setup while preserving
the same guest-visible conformance result.

The CH launcher must not silently reuse v1.4 artifacts. It requires
`artifacts/base/guest-contract.json` with the v1.5 marker and guest build
features emitted by `build-image.sh --backend ch`. This keeps CH bring-up
honest: CH packaging now comes from the v1.5 common rootfs contract, not
retargeted old CH rootfs content.

Platform-specific files in `examples/v1.5` must say why they exist. For CH,
`motlie-vmm-egress-setup.sh` and `motlie-vmm-egress.service` are explicitly
CH-only because Cloud Hypervisor exposes the egress path as a guest-visible NIC
that must be configured from per-guest overlay seed values. VZ does not install
that service because VZ egress is a host-side userspace helper.

## Implementation Order

Build v1.5 in small VMM-owned slices:

1. Create `libs/vmm/bins/v1.5/motlie-vfs-guest.rs` as the canonical v1.5
   guest mounter entrypoint.
2. Add `libs/vmm/src/guest/` for reusable guest-runtime code shared by VMM
   guest binaries and future boot services.
3. Wire the v1.5 binary in `libs/vmm/Cargo.toml`; do not wire it from
   `libs/vfs/Cargo.toml`.
4. Create `libs/vmm/examples/v1.5/` with the common image builder, common seed
   schema, common boot scripts, and one harness matrix that can run CH and VZ.
5. Consolidate manual and external-SSH validation into `harness_v1_5 shell`;
   do not carry a v1.5 REPL entrypoint forward.
6. Route harness smoke, shell, and scenario modes through one backend selector
   so the same harness binary can drive VZ locally and CH on a Linux host.
7. Build the guest Linux payload from VMM, then emit CH and VZ packaging from
   the same rootfs contract.
8. Add targeted VFS or VNET library fixes only when VMM v1.5 exposes a missing
   reusable API or a bug.
9. Keep first-contact SSH on the interactive-readiness path; put full VFS/VNET
   certification behind explicit harness validation commands.

The implementation must not create `libs/vfs/bins/v1.5`,
`libs/vfs/examples/v1.5`, `libs/vnet/bins/v1.5`, or
`libs/vnet/examples/v1.5`.

Guest binary build command:

```bash
cargo build --manifest-path libs/vmm/Cargo.toml \
  --release \
  --no-default-features \
  --features guest-vfs \
  --bin motlie-vfs-guest-v1_5
```

That command is part of the Linux guest-image build environment, not the macOS
host launcher path. `--no-default-features` is part of the contract: the guest
image build may compile the guest VFS mounter, but it must not compile or link
host runtime backends such as VNET/libslirp. The current VFS guest mounter uses
`fuser`, so the build environment must provide the FUSE development package
expected by `fuser`, and the runtime image must provide `/dev/fuse`. By default the binary requires
`/etc/motlie/v1.5/backend.env`; `--no-backend-env` is reserved for local smoke
checks that intentionally rely on defaults and process environment overrides.
The binary prints `MOTLIE_VMM_GUEST_MOUNTER_V1_5` for `--contract`; launch
paths must verify that marker and fail instead of rebuilding when it is absent.

## Common Seed And Boot Contract

The seed schema is common. Backend packaging may differ.

Seed-owned files:

```text
/etc/motlie/v1.5/backend.env
/etc/motlie-vfs/mounts.yaml
/etc/ssh/ca/user_ca.pub
/etc/ssh/auth_principals/<user>
/home/<user>/.env
```

Cloud-init owns:

- user creation
- uid/gid
- sudoers
- hostname
- SSH CA and principal material
- mount configuration
- backend environment
- service enablement or backend-specific drop-ins when required

The guest service graph is:

```text
cloud-init
  -> motlie-vfs-guest.service
  -> motlie-agent-state.service
  -> motlie-vmm-vsock-ssh.service
  -> motlie-control-plane-ready.service
```

`motlie-control-plane-ready.service` asserts interactive readiness only. It must
not run package installs, npm repair, cargo builds, package-manager quiescence
polling, egress certification, or full VFS/VNET validation.

## Success Criteria

v1.5 succeeds when:

1. `libs/vmm/examples/v1.5` owns the common CH/VZ example line.
2. CH and VZ run the same v1.5 scenarios through one harness.
3. CH and VZ use the same guest contract and seed schema.
4. The primary arm64 validation path uses the same common guest rootfs payload.
5. Guest binaries are versioned and installed through VMM-owned paths.
6. The canonical v1.5 guest mounter source lives under
   `libs/vmm/bins/v1.5`, reusable runtime code lives under
   `libs/vmm/src/guest`, and no `libs/vfs` or `libs/vnet` v1.5 bin/example
   tree is created.
7. Platform-specific code is isolated to emitters, host launchers, host
   runners/helpers, network adapters, and codesigning.
8. First-contact SSH gates on interactive readiness, not hidden build or
   certification work.
9. v1.5 has no REPL example; manual, headless, external-SSH, and scenario e2e
   validation run through `harness_v1_5`.

## Related Documents

- [`CONVERGENCE.md`](./CONVERGENCE.md)
- [`HANDOFF_VZ_TRAIT_SHAPE.md`](./HANDOFF_VZ_TRAIT_SHAPE.md)
- [`../examples/v1.35/DESIGN_IMAGE_BUILDER.md`](../examples/v1.35/DESIGN_IMAGE_BUILDER.md)
