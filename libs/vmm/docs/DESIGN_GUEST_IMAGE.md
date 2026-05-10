# motlie-vmm Guest Image And Guest Binary Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-09 | @vmm-cdx | Expand the issue #271 `mbuild` design status: app-layer CLI delegates current CH/VZ adapters, emits artifact digests, regenerates per-guest seed overlays, and delegates optional live harness validation with a validation manifest |
| 2026-05-09 | @vmm-cdx | Add the initial issue #271 product surface: `libs/vmm/examples/v1.5/motlie-image.yaml` and the standalone top-level `mbuild` builder/validator binary, currently emitting a stage manifest while package/emitter execution remains pending |
| 2026-05-09 | @vmm-cdx | Record GitHub issue #271 as a v1.5 demo success criterion: a Dockerfile-like build spec plus standalone top-level `mbuild` CLI must drive the staged image builder before the demo is accepted |
| 2026-05-09 | @vmm-cdx | Split immutable rootfs compatibility assembly from per-guest seed overlay emission; document NoCloud seed ownership, uid/gid ownership enforcement, VFS readiness waiting, dynamic backend.env sourcing, and fail-loud CH egress setup |
| 2026-05-08 | @vmm-cdx | Address PR #270 feedback in the rootfs assembler contract: install SSHD CA trust directives, make the vsock SSH loop consume backend.env, and require strict mount YAML-safe tag/path values |
| 2026-05-07 | @vmm-cdx | Add the first rootfs compatibility assembler design: mutate imported rootfs trees with v1.5 Motlie files, emit a machine-readable manifest, and record package/runtime requirements still pending for later builder/emitter stages |
| 2026-05-07 | @vmm-cdx | Tighten rootfs classifier executable probes so package-manager and systemd indicators require resolved regular files, not merely existing paths |
| 2026-05-07 | @vmm-cdx | Harden the rootfs classifier trust boundary: path reads/classification use guest-root-aware symlink resolution, escaping symlinks are rejected, and `/sbin/init` is accepted only when it resolves to systemd |
| 2026-05-07 | @vmm-cdx | Define the rootfs classifier requirements and design: stable VMM/VFS/VNET invariants stay typed in Rust while admin/profile package, mount, and init requirements are data driven |
| 2026-05-07 | @vmm-cdx | Tighten OCI whiteout safety for the rootfs importer: empty `.wh.` targets are invalid and opaque-whiteout metadata errors fail closed |
| 2026-05-07 | @vmm-cdx | Add Registry v2 platform-manifest and layer-blob fetch into a content-addressed cache that feeds importer-ready layer inputs |
| 2026-05-07 | @vmm-cdx | Add the first rootfs importer implementation slice: selected platform manifest parsing, digest-checked local layer inputs, deterministic empty assembly roots, gzip/plain tar extraction, and OCI whiteouts |
| 2026-05-07 | @vmm-cdx | Tighten resolver provenance so single-image manifests are rejected until config blob inspection verifies the requested platform |
| 2026-05-07 | @vmm-cdx | Clarify v1.5 acceptance: functional parity with v1.4 CH and v1.45 VZ through the unified v1.5 harness, image builder, and OCI-derived guest image path |
| 2026-05-07 | @vmm-cdx | Add the first OCI Registry v2 resolver implementation for image reference parsing, immutable index digest resolution, and platform manifest selection |
| 2026-05-07 | @vmm-cdx | Tighten the initial OCI implementation contract so validation records embed the typed profile and full-length SHA digests are enforced |
| 2026-05-07 | @vmm-cdx | Start the OCI import profile implementation with typed source/profile/platform/artifact metadata in `libs/vmm/src/image.rs` |
| 2026-05-06 | @vmm-cdx | Fold the CH `cloud-init.target` boot-graph fix into the OCI-derived `ubuntu-systemd` compatibility profile contract |
| 2026-05-06 | @vmm-cdx | Clarify OCI source registry and digest semantics for cross-backend validation after PR feedback |
| 2026-05-06 | @vmm-cdx | Reorder the OCI roadmap so the first contract slice starts from an Ubuntu OCI import profile and derives the Motlie guest contract from a real base image |
| 2026-05-06 | @vmm-cdx | Refine emitter timing from static-only adaptation to pre-boot persistent or ephemeral emission and document Apple VZ storage constraints |
| 2026-05-06 | @vmm-cdx | Document OCI compatibility, static emitter adaptation, external Docker image import profiles, and VFS/VNET impact for CH and VZ |
| 2026-05-05 | @vmm-cdx | Add the roadmap from a shared guest contract to per-arch OCI payloads and a final multi-arch OCI guest image artifact consumed by CH and VZ emitters |
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
- one published multi-arch OCI guest image reference once the common payload and
  emitter path are stable

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
/var/lib/cloud/seed/nocloud/user-data
/var/lib/cloud/seed/nocloud/meta-data
/etc/motlie/v1.5/backend.env
/etc/motlie-vfs/mounts.yaml
/etc/ssh/ca/user_ca.pub
/etc/ssh/auth_principals/<user>
/home/<user>/.env
```

The rootfs builder must keep those files out of the immutable common rootfs.
They are emitted per guest as NoCloud seed content or as an ephemeral/persistent
overlay applied before boot. Cloud-init consumes `user-data` and `meta-data`.
The Motlie guest services consume `backend.env`, `mounts.yaml`, SSH principal
material, and user environment files from the same per-guest seed/overlay.

Seed/overlay-owned dynamic semantics:

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

For the `ubuntu-systemd` profile, `motlie-vfs-guest.service` and
`motlie-agent-state.service` are installed under `cloud-init.target`, not
`multi-user.target`. The VFS guest mounter intentionally orders after
`cloud-final.service` because it consumes seeded user, home, and mount
configuration. Installing that unit, or a dependent unit such as
`motlie-agent-state.service`, under `multi-user.target` creates a systemd
ordering cycle on the CH path and prevents harness readiness from completing.

This rule is part of the image compatibility profile. OCI importers and CH/VZ
emitters must bake the target wiring into emitted artifacts before guest boot.
They must not rely on privileged first-contact SSH, launcher-side unit repair,
or backend-specific post-boot mutation to fix the service graph.

`motlie-control-plane-ready.service` asserts interactive readiness only. It must
not run package installs, npm repair, cargo builds, package-manager quiescence
polling, egress certification, or full VFS/VNET validation.

## Roadmap To A Shared OCI Guest Image

The end state is one OCI image reference that resolves to a multi-arch OCI image
index. That single published artifact name must describe one Motlie guest
contract, even though the contained per-arch rootfs payloads are not byte
identical.

Example target:

```text
ghcr.io/chungers/motlie-guest:v1.5
  -> linux/arm64 image
  -> linux/amd64 image
```

This is the correct convergence target for CH and VZ. A single byte-identical
rootfs blob cannot boot both Apple Silicon VZ guests and native x86_64 CH
guests because the guest userspace, packages, and kernel payloads are
architecture-specific before any boot script runs. The reusable unit is the
guest contract plus per-arch payloads, not one architecture-agnostic filesystem
blob.

The roadmap starts from a real compatible OCI base image, then generalizes the
Motlie contract from what the harness requires that base to provide.

The first supported profile should be `ubuntu-systemd` over
`docker.io/library/ubuntu:24.04`. Ubuntu is the right first foundation because
the current v1.5 harness contract already assumes an apt-based, systemd-capable
guest with sudo, OpenSSH, common Linux networking tools, and coding-agent CLI
startup checks. The tag is only the discovery input. The imported source must
be pinned by immutable OCI digests before it is accepted as a Motlie guest
profile.

The roadmap is:

1. Import a concrete Ubuntu OCI image and derive the first Motlie compatibility
   profile.
   - Resolve `docker.io/library/ubuntu:24.04` through its OCI image index.
   - Record the image reference, image-index digest, selected platform, and
     selected platform-manifest digest.
   - Select `linux/amd64` for native CH-on-DGX validation and `linux/arm64` for
     Apple Silicon VZ validation.
   - Inspect the rootfs for OS release, package manager, init system, users,
     network tooling, SSH, sudo, `/dev/fuse` assumptions, and package-manager
     state.
   - Apply the Motlie compatibility layer needed for the current v1.5 harness
     to pass.
   - Treat the required additions as the first concrete
     `ubuntu-systemd` Motlie guest contract.
   - Treat a newer Ubuntu LTS as a new profile version unless the full harness
     matrix has been rerun and the source digests have been updated explicitly.

2. Freeze the typed guest image and boot contract from that profile.
   - Define typed schemas for guest users, package baseline, systemd units,
     mount points, writable directories, SSH auto-provision behavior, and the
     validation profile.
   - Define the backend-neutral boot contract: kernel/initramfs expectations,
     seed/cloud-init ownership, required boot metadata, and guest-visible
     runtime paths.
   - CH and VZ must both be describable from this one contract even if their
     current emitters still differ.
   - Keep Ubuntu-specific requirements in the `ubuntu-systemd` profile instead
     of confusing them with the whole VMM image contract.

3. Build one common rootfs assembler.
   - Move guest payload assembly behind one VMM-owned builder that installs the
     shared package set, services, Motlie guest binaries, Codex/git/sudo
     contract, and validation markers.
   - Keep architecture-specific payload bits explicit in the builder. This
     includes Linux packages and any arch-specific Motlie guest binaries.
   - The output of this stage is a canonical assembled rootfs tree plus a
     manifest with contract version and payload checksums.

4. Publish per-arch OCI guest images from the common rootfs assembly.
   - Treat OCI as the canonical distribution format for the guest payload, not
     as a hypervisor-native boot format.
   - Publish at least one OCI image per guest architecture:
     `motlie-guest:arm64` and `motlie-guest:amd64`.
   - Embed contract version, validation profile, and provenance labels so both
     backend emitters consume the same payload definition.

5. Make CH and VZ emitters consume the OCI payload.
   - CH emitter: OCI payload -> CH boot artifacts.
   - VZ emitter: OCI payload -> VZ boot artifacts.
   - Backend-specific packaging remains allowed here, but it must be derived
     from the same OCI payload and the same typed contract instead of from
     backend-specific guest build scripts.
   - This is the stage that gives true cross-backend image reuse in the product
     sense: one source payload, two backend emitters.

6. Require one harness validation matrix across both backends for the same OCI
   image-index digest and declared platform variants.
   - The saved scenarios for bootstrap, SSH auto-provision, VFS, VNET/egress,
     PTY/Codex, and multi-guest operation must pass for CH and VZ from the same
     logical image contract.
   - Validation results should record the OCI image-index digest, selected
     platform, selected platform-manifest digest, contract version, backend
     kind, emitted backend artifact digests, and emitted artifact metadata.

7. Publish one multi-arch OCI image reference.
   - After the per-arch OCI payloads and emitters are stable, publish one OCI
     image index / manifest list as the canonical guest image reference.
   - That one reference is the final externally visible artifact for the guest
     image line. CH and VZ emitters select the correct per-arch payload from
     the index and produce backend-local boot artifacts from it.

This roadmap intentionally stops short of requiring one byte-identical final VM
boot file for both backends. That stronger convergence would additionally
require:

- one shared guest architecture
- one shared disk boot model
- one shared partition and bootloader layout

Those constraints do not hold for Apple Silicon VZ and native x86_64 CH. The
right end state here is one logical guest contract and one multi-arch OCI image
artifact, not one architecture-agnostic raw boot image.

### OCI Compatibility Model

OCI is the distribution format for the guest payload, not the native VM boot
format. The OCI image specification defines image manifests, image indexes, and
layers. The index is the multi-platform object: it points to image manifests for
specific operating system and architecture variants. Docker documents the same
model as a manifest list where one image name selects the correct platform
variant at pull time.

References:

- OCI image specification: image manifest, image index, and layer model:
  `https://oci-playground.github.io/specs-latest/specs/image/v1.0.0/oci-image-spec.html`
- OCI layer specification: layers are filesystem changesets with additions,
  modifications, removals, and whiteouts:
  `https://raw.githubusercontent.com/opencontainers/image-spec/main/layer.md`
- Docker multi-platform images: one image name can resolve to per-platform
  manifests such as `linux/amd64` and `linux/arm64`:
  `https://docs.docker.com/build/building/multi-platform/`

The VMM image builder must therefore treat OCI as:

```text
OCI image index
  -> platform manifest
  -> rootfs layers + config
  -> Motlie compatibility adaptation
  -> backend emitter
  -> CH or VZ boot artifacts
```

The OCI payload does not remove the need for backend emitters. The emitters
still own kernel/initramfs or disk boot metadata, seed/cloud-init material,
runtime overlays, launch manifests, and platform-specific host artifacts.

### Pre-Boot Emitter Adaptation Versus Post-Boot Mutation

Emitter adaptation happens before the VM boots into the guest. It may be
persistent or ephemeral. The builder resolves an OCI platform variant, assembles
the rootfs, applies a Motlie compatibility layer, verifies the guest contract,
and emits backend boot artifacts before handing control to CH or VZ.

Pre-boot emitted artifacts may live in:

- a persistent artifact directory for reuse and inspection
- a temporary directory under the harness runtime root
- tmpfs or another RAM-backed filesystem when the host/backend supports it
- a backend-specific file-handle path when the backend API accepts file handles

Pre-boot adaptation owns:

- installing Motlie guest binaries under `/opt/motlie/v1.5/guest/bin`
- exposing compatibility paths under `/usr/local/bin`
- installing systemd units or an approved Motlie init profile
- enabling the common boot graph under `cloud-init.target` for the
  `ubuntu-systemd` profile, including `motlie-vfs-guest.service` and
  `motlie-agent-state.service`
- creating stable mount point directories such as `/workspace`,
  `/agent-state`, and `/home/<user>` templates where the image policy requires
  them
- installing baseline packages required by the selected validation profile
- creating backend-neutral config directories and service defaults; concrete
  `backend.env` and `mounts.yaml` contents are per-guest seed/overlay files
- enabling the CH egress setup service only in CH-emitted artifacts when CH
  requires guest-side route/DNS programming

Dynamic boot-time configuration should be limited to per-guest state:

- username, uid, gid, hostname, and SSH principal material
- per-guest mount declarations and host-backed mount identities
- per-guest egress addresses, MACs, and route/DNS values where the backend
  requires them
- runtime socket paths and control-plane readiness files

Privileged SSH or TTY mutation after boot is allowed only as a diagnostic or
transitional importer path. It must not be the normal product contract because
it requires the guest to already be bootable, reachable, privileged, and
partially configured. If an OCI image needs Motlie packages, units, mount
points, or network tooling, the supported path is to adapt the image before
guest boot, whether that adapted artifact is persistent or ephemeral.

The hard product boundary is not "disk versus RAM." It is "before guest boot
versus after guest boot." On-demand emission during harness startup is valid if
the guest sees a complete Motlie contract when it starts.

### Apple VZ Storage And Ephemeral Artifact Constraints

Apple Virtualization.framework supports path-backed and file-handle-backed
storage, but the concrete APIs matter when deciding whether an emitter can use
tmpfs, memfd, or a normal disk file.

Relevant Apple APIs:

- `VZDiskImageStorageDeviceAttachment` attaches a disk image by URL. Apple
  documents support for RAW and ASIF disk images, and the initializer takes a
  local file URL.
- `VZDiskBlockDeviceStorageDeviceAttachment` attaches an actual disk through a
  file handle.
- `VZLinuxBootLoader` takes kernel and optional initial RAM disk URLs.
- `VZFileHandleNetworkDeviceAttachment` accepts a file handle for a connected
  datagram socket. This supports the current VZ userspace egress shape, but it
  is a network-device API, not a disk-image API.

References:

- Apple `VZDiskImageStorageDeviceAttachment`:
  `https://developer.apple.com/documentation/virtualization/vzdiskimagestoragedeviceattachment`
- Apple `VZDiskBlockDeviceStorageDeviceAttachment`:
  `https://developer.apple.com/documentation/virtualization/vzdiskblockdevicestoragedeviceattachment`
- Apple `VZLinuxBootLoader`:
  `https://developer.apple.com/documentation/virtualization/vzlinuxbootloader`
- Apple `VZFileHandleNetworkDeviceAttachment`:
  `https://developer.apple.com/documentation/virtualization/vzfilehandlenetworkdeviceattachment`

Implications for VZ:

- A VZ emitter can generate RAW or ASIF disk artifacts on demand before launch.
- A normal local file path is the most compatible VZ storage target because
  `VZDiskImageStorageDeviceAttachment` is URL-based.
- RAM-backed path storage is acceptable if the host exposes it as a normal
  local file path that VZ can open for the lifetime of the VM.
- Linux `memfd` is not a portable VZ artifact target by itself. It has no stable
  macOS equivalent, and the URL-based disk-image API expects a file path.
- A file-handle storage path may be possible through
  `VZDiskBlockDeviceStorageDeviceAttachment`, but that is a different backend
  contract and should be designed explicitly before becoming the default.

Implications for CH:

- CH can support on-demand pre-boot emission to persistent files, temporary
  files, or Linux tmpfs-backed files.
- Raw `memfd` may be possible only if the CH launcher path can pass a usable
  file descriptor or `/proc/self/fd/...` path for the specific artifact. The
  conservative portable path is still a normal file path, possibly on tmpfs.

The default cross-backend emitter contract should therefore be path-oriented:
emit a complete backend artifact set under a managed artifact root, where that
root may be persistent or ephemeral. Backend-specific file-descriptor storage is
an optimization, not the base contract.

The demo mount points and package choices in `examples/v1.1` through `v1.5`
are product-level example configuration. They should not become global VMM
defaults. The reusable contract is the mechanism: image profiles declare which
paths, packages, users, services, and validation scenarios are required for a
particular product guest.

### VFS And VNET Compatibility With OCI Payloads

OCI payloads do not break VFS memfs semantics. `motlie-vfs` controls only paths
inside trees mounted through its FUSE/vsock/RPC client; it does not mutate
arbitrary rootfs paths outside those mounts. Memfs layers already sit above a
base layer and can add content, synthetic directories, and whiteouts. The
compatibility requirement is that the booted image provides the guest-side
mount runner, `/dev/fuse`, mount configuration, and the target mount points
required by the selected profile.

If an imported OCI image already contains meaningful files under a path that
Motlie later mounts over, the mount hides the underlying rootfs path. That is
normal Linux mount behavior and must be handled by the image profile. Profiles
can reserve those paths, import selected lower-rootfs content into the VFS base
layer, or choose different mount points.

OCI payloads also do not break VNET/libslirp egress. The egress service is
host-side. The guest contract is limited to the network device, route, DNS, and
tooling needed to use that egress path. CH may need a guest-side service that
finds the egress NIC and programs route/DNS state. VZ may obtain route/DNS from
the VZ userspace egress path. Both remain compatible with the same OCI-derived
rootfs if the selected profile installs the required guest-side tooling.

### External Docker/OCI Image Import Profiles

Already available OCI images can be useful foundations, but they must pass an
import profile before VMM treats them as Motlie guest images.

Good initial candidates:

- `docker.io/library/ubuntu:24.04` for the first `ubuntu-systemd` profile.
  Docker Hub publishes Ubuntu as a Docker Official Image, and Canonical
  documents Ubuntu OCI images as multi-architecture images built from minimal
  rootfs tarballs. Newer Ubuntu LTS tags require a new profile version or an
  explicit digest update with the full harness matrix rerun.
- `alpine:3` or another supported Alpine tag for a smaller experimental
  profile. Docker Hub publishes Alpine as a Docker Official Image with
  `amd64` and `arm64v8` support, among other architectures.

References:

- Ubuntu Docker Official Image:
  `https://hub.docker.com/_/ubuntu`
- Ubuntu OCI image configuration, including minimal rootfs layers and
  multi-architecture image index:
  `https://documentation.ubuntu.com/oci-registries/oci-reference/oci-image-configuration/`
- Alpine Docker Official Image, including supported architectures:
  `https://hub.docker.com/_/alpine`

These images are foundations, not ready Motlie VM guests. Most container images
are optimized for a container runtime and may not include systemd, sshd, sudo,
cloud-init, `/dev/fuse` assumptions, DHCP/network tooling, or package-manager
state suitable for a VM boot contract.

The importer must classify each source image:

```text
ExternalOciSource
  image_ref
  image_index_digest
  platform
  platform_manifest_digest
  os_release
  libc_family
  package_manager
  init_profile
  motlie_compatibility_profile
```

For the first profile, `image_ref` is `docker.io/library/ubuntu:24.04`. The
profile must store the resolved image-index digest and the selected platform
manifest digest. Tags are not sufficient for reproducible validation because
registries can move them.

Supported import profiles:

- `ubuntu-systemd`: apt-based, systemd-capable, expected to support the full
  v1.5 agent validation profile after pre-boot adaptation.
- `alpine-openrc` or `alpine-motlie-init`: apk-based, smaller profile, likely
  requires a Motlie-owned init service path instead of assuming the current
  systemd service graph.
- `unsupported`: image can be unpacked for inspection, but cannot be emitted as
  a Motlie guest without a profile that defines init, SSH, VFS, VNET, and
  validation behavior.

The first implementation should use Docker Hub's Ubuntu official image because
the current v1.5 examples already assume apt, `sudo -n apt-get update`, systemd
units, and coding-agent CLI startup checks. Alpine is feasible, but it is a
separate profile because package names, init system, shell/coreutils behavior,
and service management differ. The Motlie contract should emerge from the
Ubuntu profile first, then be factored into profile-independent requirements
and profile-specific requirements once a second base image is implemented.

### Current Implementation Slice

`libs/vmm/src/image.rs` is the first Rust surface for this roadmap. It resolves
registry manifest metadata, fetches selected platform manifests and layer blobs
into a content-addressed cache, parses selected platform manifests, unpacks
digest-checked rootfs layer blobs into a deterministic assembly root, classifies
that rootfs, and applies the first mutation-only Motlie compatibility layer. It
establishes the typed metadata that classifier, assembler, emitter, harness, and
CI code must share. It does not yet run package-manager installation and it does
not yet emit VM boot artifacts.

- `OciPlatform` and `GuestArchitecture` record the selected OCI platform.
- `OciDigest` records immutable image-index, platform-manifest, and emitted
  artifact digests.
- `OciImageReference` parses and normalizes Docker/OCI-style image references.
- `ExternalOciSource` records the source image reference plus resolved
  immutable OCI identity.
- `GuestImageProfile` records the Motlie compatibility profile derived from
  the external source, starting with `ubuntu-systemd`.
- `GuestImageValidationRecord` embeds the typed `GuestImageProfile`, then
  records backend kind, contract version, and emitted artifact digests needed
  to prove which guest image/profile combination was validated.
- `OciRegistryClient` resolves an image reference through Registry v2,
  including Bearer auth challenge handling, local `sha256` digest computation
  for the returned index/manifest body, optional `Docker-Content-Digest`
  verification, and selected platform-manifest digest extraction from an OCI
  image index or Docker manifest list. Single-image manifests are rejected until
  the resolver fetches the manifest config blob and verifies `os` /
  `architecture`.
- `OciContentCache` stores selected platform manifests and layer blobs under
  `blobs/<algorithm>/<encoded>`. Cache hits are revalidated by digest and, for
  layer descriptors, expected size. Corrupt cache entries fail closed rather than
  being silently overwritten.
- `OciRegistryClient::fetch_resolved_platform_to_cache(...)` fetches the selected
  platform manifest by immutable digest, parses it, fetches all rootfs layer
  blobs by digest, validates `Docker-Content-Digest` when present, and returns
  importer-ready `OciLayerInput` values pointing at verified cached content.
- `OciPlatformManifest` parses the selected platform manifest into config digest
  and layer descriptors.
- `OciRootfsImporter` applies caller-provided local layer blobs after validating
  descriptor size and digest. The importer requires an empty assembly root,
  supports OCI plain tar, OCI gzip tar, and Docker gzip rootfs diff layers, and
  applies OCI whiteout and opaque-directory semantics. The first implementation
  supports `sha256` layer descriptors only; other digest algorithms must be added
  deliberately rather than silently producing a mismatched hash. Empty `.wh.`
  targets are invalid, and opaque-whiteout metadata errors other than `NotFound`
  fail closed instead of reporting a successful import with stale lower-layer
  content.
- `RootfsProfileSpec` and `RootfsClassifier` classify an imported rootfs before
  mutation. The built-in `ubuntu-systemd` spec is derived from
  `GuestImageProfile`, but package, mount, OS, init, binary, VFS, and VNET
  requirements remain profile data so production builders can change them
  without changing classifier code. Results are typed as `ready`,
  `compatible-with-adaptation`, or `unsupported`, with machine-readable findings.
  Classifier path access is rootfs-safe: it uses symlink metadata, resolves
  symlinks as guest paths, treats absolute symlink targets as guest-root-relative,
  rejects parent traversal above the guest root, and caps symlink recursion.
  `/sbin/init` is not a standalone systemd indicator; it is accepted only when it
  resolves inside the rootfs to `/usr/lib/systemd/systemd` or
  `/lib/systemd/systemd`. Package-manager and systemd indicator probes require
  resolved regular files; directories or other non-file path types are not
  accepted as present.
- `RootfsCompatibilityAssembler` applies the immutable Motlie compatibility
  layer to a supported imported rootfs. It installs v1.5 guest payloads under
  `/opt/motlie/v1.5/guest/bin`, exposes compatibility symlinks under
  `/usr/local/bin`, installs the `ubuntu-systemd` service graph, creates stable
  directories, installs SSHD CA trust directives, and emits
  `RootfsCompatibilityAssemblyManifest` with installed paths and pending
  requirements. It does not write per-guest `backend.env`, `mounts.yaml`, SSH
  principals, sudo/user env files, or cloud-init seed files.
- `RootfsSeedOverlayAssembler` emits the per-guest seed/overlay layer. It writes
  NoCloud `user-data` / `meta-data`, `backend.env`, `mounts.yaml`, SSH CA and
  principal material, sudoers, and user `.env` files. User seeds require uid/gid
  so cloud-init can create the OS user and seed-owned files such as
  `/home/<user>/.env` can be chowned before boot. Missing installable packages
  or missing systemd remain explicit manifest evidence unless the caller selects
  the fail-fast pending-requirement policy; this slice does not fake package
  installation.

### Rootfs Classifier Requirements And Design

The image-builder gate after OCI import is a non-mutating rootfs classifier. It
inspects an imported OCI rootfs and answers whether that rootfs is a supported
foundation for the selected Motlie guest profile before any Motlie compatibility
layer is applied.

Functional requirements:

- The classifier input is an imported rootfs path plus a typed
  `RootfsProfileSpec`.
- `RootfsProfileSpec` is data driven. It is derived from `GuestImageProfile` for
  the built-in `ubuntu-systemd` profile, but admins and production builders can
  provide different package, mount-point, distro/version, and init requirements
  without changing classifier code.
- The classifier must not bake v1.5 example choices such as demo users,
  demo-only packages, or demo mount layout as product invariants.
- Stable Motlie/VMM invariants stay typed in Rust: safe absolute rootfs paths,
  Linux filesystem shape, VFS mount-point and `/dev` assumptions, FUSE runtime
  device expectation, and VNET/egress configuration path assumptions.
- Profile requirements are explicit data: accepted OS release IDs/versions,
  init profile, required packages, required binaries, required mount points, and
  later required users/services.
- Classification is read-only. It reports what exists, what can be installed by
  the compatibility layer, what must be provided by runtime provisioning, and
  what is unsupported.
- Path classification and file reads must never follow host filesystem symlinks
  out of the imported assembly root. Valid guest symlinks, including absolute
  guest-root-relative links such as `/usr/lib/os-release`, are supported.
- The result must be machine-readable for harness and CI use. It must include an
  overall status plus typed findings with requirement kind, status, path/package
  where relevant, and evidence.

The first status model is:

- `ready`: every requirement inspected from the rootfs is already present.
- `compatible-with-adaptation`: the rootfs is a supported foundation, but needs
  pre-boot Motlie compatibility work, package installation, mount directory
  creation, or runtime device provisioning.
- `unsupported`: at least one stable invariant or selected profile requirement
  cannot be satisfied safely by the builder.

This keeps the classifier general. For example, the v1.5 example profile may
require `git`, `openssh-server`, and `/workspace`, while a production profile may
choose a different package set and mount layout. Both profiles still share the
same VFS/VNET invariants and the same typed classification result.

The current `ubuntu-systemd` profile validates against
`docker.io/library/ubuntu:24.04` and `InitProfile::UbuntuSystemd`; validation
must fail if a caller combines the Ubuntu profile name with a different source
image or init profile. SHA-family OCI digests must be full-length digests, not
short placeholders, because validation records are provenance artifacts.

### Rootfs Compatibility And Seed Overlay Requirements And Design

The immutable compatibility assembler is the first mutating step after rootfs
import and classification. Its job is to turn a supported foundation rootfs into
a reusable Motlie v1.5 rootfs contract before any CH or VZ backend emitter
packages it. Per-guest state is a separate seed/overlay emission step.

Functional requirements:

- The assembler input is an imported rootfs path plus a typed
  `RootfsCompatibilityLayerSpec`.
- The assembler must classify the rootfs first and refuse `unsupported`
  foundations. Mutation only begins after the read-only classifier reports a
  supported foundation.
- The compatibility assembler installs immutable Motlie contract files only:
  guest binaries, support scripts, systemd units, compatibility symlinks,
  profile scripts, SSHD CA trust directives, backend-neutral directories, and
  required mount-point directories.
- The compatibility assembler must not write per-guest state into the common
  rootfs: no `backend.env`, no `mounts.yaml`, no NoCloud seed, no SSH
  principals, no sudoers, and no user `.env` files.
- `RootfsSeedOverlayAssembler` owns the per-guest seed/overlay layer:
  NoCloud `user-data` and `meta-data`, `backend.env`, `mounts.yaml`, SSH CA key,
  auth principals, sudoers, and user `.env` files.
- User seed entries require uid/gid. The seed overlay renders cloud-init user
  creation and also applies the same uid/gid ownership to user-owned seed files,
  preventing root-owned `0600` files from becoming unreadable after boot.
- The assembler must not run apt, dpkg, systemctl, chroot, SSH, TTY, or any
  backend-specific emitter operation. Package-manager execution belongs to a
  later builder/package strategy. Backend artifact packaging belongs to CH/VZ
  emitters.
- Missing installable requirements are explicit data. The default policy records
  them in `RootfsCompatibilityAssemblyManifest.pending_requirements`; a stricter
  caller can fail on installable pending requirements.
- The built-in `ubuntu-systemd` package baseline mirrors the current v1.5
  validation image assumptions, including systemd/cloud-init, OpenSSH/sudo,
  VFS/FUSE support, networking/debugging tools, `socat` for the vsock SSH loop,
  and coding-agent CLI prerequisites such as `git` and `npm`. Production
  builders can supply a different package set through `RootfsProfileSpec`.
- SSH CA auto-provisioning is split: the common rootfs installs the OpenSSH
  drop-in at `/etc/ssh/sshd_config.d/90-motlie-vmm-ca.conf` and an empty
  `/etc/ssh/ca/user_ca.pub` placeholder, while the seed overlay supplies the
  per-guest CA key contents and `/etc/ssh/auth_principals/<user>`.
- The vsock SSH loop re-sources `/etc/motlie/v1.5/backend.env` inside its retry
  loop so seed/overlay refreshes are visible without relying on stale shell
  variables. The default remains port `2222`.
- `motlie-agent-state-setup` waits for `/agent-state` and the target home
  directory to appear in `/proc/mounts` before bind-mounting agent state into
  the guest home. The unit graph therefore does not depend on `Type=notify`,
  but the script has an explicit readiness contract rather than only
  `After=motlie-vfs-guest.service`.
- CH egress setup must fail loudly if the selected egress interface cannot be
  brought up. Skipping route/DNS configuration after a failed `ip link set up`
  is not a successful boot contract.
- Mount config rendering stays manual for this first slice, but it is not
  freeform: mount tags and mount guest-path components must be ASCII tokens
  using only letters, digits, `_`, `-`, or `.`. Guest mount paths must be
  absolute, non-root paths. This keeps the YAML surface deterministic until a
  later typed config serializer is introduced.
- Runtime requirements such as `/dev/fuse` stay visible as pending runtime
  provisioning rather than being hidden by rootfs mutation.
- All writes must stay inside the imported rootfs. Parent path symlinks are
  rejected during writes so a hostile or malformed OCI rootfs cannot redirect
  assembler output into the host filesystem.
- The manifest must be machine-readable and sufficient for later emitters and
  harness/CI to explain what was installed and what remains pending.

The first API surface is:

```rust
let mut rootfs_spec = RootfsCompatibilityLayerSpec::new(rootfs_profile_spec);
rootfs_spec.guest_binaries.push(
    RootfsPayloadFile::new(host_guest_binary, MOTLIE_V15_GUEST_BIN_OPT, 0o755)
        .with_link(MOTLIE_V15_GUEST_BIN_COMPAT),
);

let rootfs_manifest = RootfsCompatibilityAssembler::new().assemble(rootfs, &rootfs_spec)?;

let mut seed_spec = RootfsSeedOverlaySpec::new(
    RootfsCloudInitSeed::new("alice", "motlie-alice"),
    RootfsCompatibilityBackendEnv::for_backend("ch", "ch-vhost-user"),
);
seed_spec.mounts.push(RootfsMountSpec::new("alice-workspace", "/workspace"));
let mut alice = RootfsUserSeed::new("alice", "alice");
alice.uid = Some(1000);
alice.gid = Some(1000);
seed_spec.users.push(alice);

let seed_manifest = RootfsSeedOverlayAssembler::new().assemble(seed_overlay_root, &seed_spec)?;
```

The manifest is the handoff contract for the next slices:

- Package strategy consumes `pending_requirements` and either installs the
  package/profile baseline or rejects the image before emit.
- CH/VZ emitters consume the mutated rootfs plus installed-path metadata and add
  only backend artifact packaging, seed overlays, kernel/disk metadata, and
  backend-specific launch inputs.
- Harness validation records consume the source profile plus emitted artifact
  digests and prove v1.5 parity against CH and VZ.

The current helper
`OciPlatform::default_for_v1_5_validation_backend(BackendKind)` is only a lab
default: CH validation currently targets native amd64 DGX hosts and VZ
validation currently targets Apple Silicon arm64 hosts. The selected platform
remains explicit in `ExternalOciSource` and must be supplied by callers when a
different CH or VZ host architecture is used.

### Dockerfile-Like Builder Contract For v1.5 Demo

GitHub issue #271 is the tracking issue for the durable v1.5 image-builder
contract. PR #270 introduces lower-level implementation stages plus the first
checked-in config/CLI surface. It must not be read as making Rust structs the
product interface. The v1.5 demo is accepted only after #271 is closed by the
config-driven builder running the full staged image flow.

Required product surface:

- checked-in Dockerfile-like config:
  `libs/vmm/examples/v1.5/motlie-image.yaml`
- ordered stages for source resolve, import, classify, package install,
  immutable Motlie layer, image policy, seed overlay, backend emission, and
  validation
- explicit immutable image files versus per-guest seed files
- package-manager stages for apt first, with room for apk/dnf/zypper/pacman
  profiles without changing the core lifecycle
- sshd image policy for CA trust, ForceCommand/on-login hooks, and reusable
  hardening; dynamic CA key and principals stay seed data
- CH and VZ emitter targets from the same immutable rootfs contract
- validation requirements for VFS memfs views, egress/package-index refresh,
  Codex startup, and Claude startup

The standalone binary is the durable operator/CI entrypoint:

```sh
mbuild build --config libs/vmm/examples/v1.5/motlie-image.yaml --target ch --out artifacts/v1.5/ch
mbuild build --config libs/vmm/examples/v1.5/motlie-image.yaml --target vz --out artifacts/v1.5/vz
mbuild seed --config libs/vmm/examples/v1.5/motlie-image.yaml --target ch --guest alice --uid 2001 --gid 2001 --out artifacts/v1.5/seed/alice
mbuild validate --config libs/vmm/examples/v1.5/motlie-image.yaml --artifact artifacts/v1.5/ch --require-executed --scenario libs/vmm/examples/v1.5/scenarios/multiguest-validate.json
```

`RootfsClassifier`, `RootfsCompatibilityAssembler`, and
`RootfsSeedOverlayAssembler` are builder stages behind that config/CLI
contract. Downstream CH/VZ emitters should consume stage manifests from the
builder, not reconstruct the lifecycle from example shell scripts.

Current binary location:

```text
bins/mbuild/src/main.rs
```

`mbuild build` consumes the config and writes `mbuild-manifest.json` with
source/import/classify/package/immutable-layer/policy/backend-emitter evidence,
delegating to the current v1.5 CH/VZ adapters while they remain transitional.
`mbuild seed` writes `mbuild-seed-manifest.json` and regenerates per-guest
NoCloud/backend/VFS/SSH/user seed files without rebuilding the immutable image.
`mbuild validate --scenario` delegates live conformance to `harness_v1_5` and
writes `mbuild-validation-manifest.json` with the harness command, log path,
scenario, target, and exit status.

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
10. The guest-image roadmap is explicit: a shared typed contract, a common
    rootfs assembler, per-arch OCI guest payloads, backend emitters that
    consume those payloads, and a final published multi-arch OCI guest image
    reference.
11. GitHub issue #271 is closed by the v1.5 demo: a Dockerfile-like checked-in
    builder spec and standalone top-level `mbuild` binary drive the image build,
    validation, and machine-readable stage manifests.

## v1.5 Functional Parity Acceptance

The v1.5 line is accepted only when it is functionally equivalent to the
historical v1.4 CH path and v1.45 VZ path while using the unified v1.5
implementation shape:

- one v1.5 harness entrypoint and scenario matrix
- one VMM-owned image builder contract
- one OCI-derived guest image/profile flow
- backend-specific emitters only for CH/VZ boot-artifact and launcher
  differences

The historical lines remain acceptance baselines, not the implementation
target. v1.5 must preserve their guest-visible behavior:

- multi-guest lifecycle and isolation
- SSH first-contact and auto-provisioning
- VFS mounts for home/workspace/agent-state semantics
- VNET/egress DNS, HTTPS, and apt readiness
- passwordless sudo where the validation profile requires it
- PTY/TUI interaction, including Codex/Claude startup checks with no OS-level
  execution errors
- reproducible harness artifacts sufficient for later debugging

Resolver and OCI-import tests are necessary sub-gates, not sufficient final
acceptance. A resolver PR must run live registry validation when it changes
source resolution. An importer/emitter PR must additionally prove that the
resolved OCI payload flows through the unified image builder into CH and VZ
artifacts. Final v1.5 acceptance requires the shared v1.5 harness matrix to
pass against both CH and VZ from the OCI-derived guest image path.

## Related Documents

- [`CONVERGENCE.md`](./CONVERGENCE.md)
- [`HANDOFF_VZ_TRAIT_SHAPE.md`](./HANDOFF_VZ_TRAIT_SHAPE.md)
- [`../examples/v1.35/DESIGN_IMAGE_BUILDER.md`](../examples/v1.35/DESIGN_IMAGE_BUILDER.md)
