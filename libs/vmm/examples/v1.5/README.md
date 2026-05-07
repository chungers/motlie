# motlie-vmm v1.5 Common Guest Image

`v1.5` is the common CH/VZ convergence line. It must demonstrate the same
capabilities as CH `v1.4` and VZ `v1.45` using one VMM-owned guest contract,
one seed schema, one service graph, and one harness matrix.

The source of truth is [`../../docs/DESIGN_GUEST_IMAGE.md`](../../docs/DESIGN_GUEST_IMAGE.md).

## Ownership

- VMM owns this directory, the guest image builder, seed schema, boot scripts,
  harness, and guest binary packaging.
- Both backend paths live here: `launch-ch.sh` for Cloud Hypervisor and
  `launch-vz.sh` for Apple VZ. Shared shell constants/rendering helpers live in
  `common-contract.sh`.
- The canonical v1.5 guest mounter source lives at
  `libs/vmm/bins/v1.5/motlie-vfs-guest.rs`.
- Reusable guest runtime code lives under `libs/vmm/src/guest`.
- Do not add `libs/vfs/bins/v1.5`, `libs/vfs/examples/v1.5`,
  `libs/vnet/bins/v1.5`, or `libs/vnet/examples/v1.5`.
- VFS/VNET changes during this work must be limited to reusable library bug
  fixes or functionality-gap fixes.

## Builder And Launcher Layout

`examples/v1.5` is intentionally the only example directory for this phase:

```text
common-contract.sh          shared guest-visible constants and render helpers
build-image.sh              common builder entrypoint; dispatches backend emitters
build-guest.sh              current VZ packaging path for the common payload
build-ch-artifacts.sh       Linux CH emitter for rootfs.squashfs/kernel contract
launch-ch.sh                CH launcher/adaptation, consumes artifacts/base
launch-vz.sh                VZ launcher/adaptation, consumes native VM artifacts
harness/                    single CH/VZ-directed harness and shell surface
backend.env.example         common backend.env schema
backend.env.ch.example      CH values for the common schema
backend.env.vz.example      VZ values for the common schema
motlie-*.service            common guest service graph
motlie-vmm-egress.*         CH-only egress NIC adaptation, explicitly marked
scenarios/*.json            common harness scenario contract
```

The split above is the intended Rust extraction seam. `build-image.sh` is the
temporary shell CLI for that future Rust builder. Common schema, contract
markers, guest binary paths, and seed rendering should become typed Rust data.
Backend-specific code should become CH and VZ emitters/runners that consume the
same typed contract.

`repl_host_v1_5` is deliberately excluded. Any capability needed for manual or
external-SSH e2e validation belongs in `harness_v1_5 shell`, including
interactive auto-provision toggling.

`harness_v1_5` has an explicit `--backend vz|ch` selector. VZ is the validated
local backend today. CH is wired as the same harness surface and expects
`build-image.sh --backend ch` to emit CH base artifacts under `artifacts/base`
before it can run on a Linux CH host.

`launch-ch.sh` and `launch-vz.sh` are not allowed to build guest binaries,
install packages, repair npm/rustup state, or mutate immutable guest services.
They may only wire platform-specific host artifacts and stage dynamic
per-guest seed data.

## Guest Binary

The guest mounter binary is built from VMM:

```bash
cargo build --manifest-path libs/vmm/Cargo.toml \
  --release \
  --no-default-features \
  --features guest-vfs \
  --bin motlie-vfs-guest-v1_5
```

The image builder installs it as:

```text
/opt/motlie/v1.5/guest/bin/motlie-vfs-guest
/usr/local/bin/motlie-vfs-guest
```

This build belongs in the Linux guest-image build environment. The current
guest mounter uses `fuser`, so the build environment needs FUSE development
packages and the runtime image needs `/dev/fuse`.

By default the binary requires `/etc/motlie/v1.5/backend.env`. Use
`--no-backend-env` only for local parser or smoke checks that intentionally rely
on built-in defaults and process environment overrides.

Image builds must compile this binary once and install it into the base image.
Guest boot and `launch-vz.sh` must not rebuild it; they only verify the
`MOTLIE_VMM_GUEST_MOUNTER_V1_5` contract marker and fail fast if the image is
stale.

For CH, run the common builder entrypoint from the repository root on the CH
host:

```bash
./libs/vmm/examples/v1.5/build-image.sh --backend ch
```

This produces:

```text
libs/vmm/examples/v1.5/artifacts/base/rootfs.squashfs
libs/vmm/examples/v1.5/artifacts/base/Image|vmlinux.bin
libs/vmm/examples/v1.5/artifacts/base/guest-contract.json
```

The CH emitter builds the same VMM-owned guest mounter once and installs the
same guest-visible paths. CH-specific differences are limited to kernel/rootfs
packaging, runtime overlay creation, and `motlie-vmm-egress.*` NIC setup. On
macOS, `build-image.sh` defaults to the current VZ packaging emitter.

## Seed Files

The common seed must provide:

```text
/etc/motlie/v1.5/backend.env
/etc/motlie-vfs/mounts.yaml
/etc/ssh/ca/user_ca.pub
/etc/ssh/auth_principals/<user>
/home/<user>/.env
```

`backend.env.example` and `mounts.example.yaml` are schema examples for the
first v1.5 image-builder slice.

## Service Graph

The converged boot graph is:

```text
cloud-init
  -> motlie-vfs-guest.service
  -> motlie-agent-state.service
  -> motlie-vmm-vsock-ssh.service
  -> motlie-control-plane-ready.service
```

`motlie-vfs-guest.service` is installed under `cloud-init.target`, not
`multi-user.target`. The guest mounter depends on `cloud-final.service` for the
seeded user/home contract, so enabling it under `multi-user.target` creates an
ordering cycle and blocks first-contact readiness on the CH path. Dependent
units that order themselves after `motlie-vfs-guest.service`, such as
`motlie-agent-state.service`, must stay on the same `cloud-init.target` path.
This is now part of the OCI-derived `ubuntu-systemd` compatibility profile:
builders and backend emitters must bake the target wiring before guest boot
instead of repairing units through first-contact SSH or launcher mutation.

First-contact SSH must wait only for interactive readiness. Full egress, CLI,
package-manager, and VFS/VNET certification belongs in explicit harness
validation, not in the first SSH path.
