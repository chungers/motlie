# motlie-vmm: Apple Vz Guest Image Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-14 | @codex-vz | Initial design for the Apple Vz guest image pipeline: define the boot artifacts for `vz-runner`, compare image-build options, state the aarch64 Linux requirements for `VZLinuxBootLoader` + `VZGenericPlatformConfiguration`, and define how Vz guest images relate to the existing CH image contract |

## Goal

Define the guest image pipeline required to boot a Linux guest under Apple
`Virtualization.framework`.

This is a prerequisite for the Vz filesystem and networking PoCs because:

- `#172` cannot validate guestfs without a bootable aarch64 Linux guest
- the SSH/auto-provision path in `libs/vmm` also depends on the guest boot and
  cloud-init contract being stable

The design target is not “reuse Cloud Hypervisor artifacts unchanged.” The
target is:

- preserve the same guest userspace contract where possible
- accept that Apple Vz wants different boot artifact packaging
- make the Vz image path reproducible enough for both local development and CI

## Requirements

The phase-1 Vz guest image path must provide:

- an aarch64 Linux kernel suitable for `VZLinuxBootLoader`
- an optional initrd/initramfs if the chosen guest layout needs one
- a writable root disk image attached as a block device
- a separate NoCloud cloud-init disk image
- guest userspace contents compatible with the CH line where parity matters:
  - OpenSSH server
  - `cloud-init`
  - `motlie-vfs-guest`
  - guest services such as `motlie-vmm-vsock-ssh`
  - the same CA trust / user-provisioning semantics used by `v1.4`

## What `vz-runner` Needs To Boot

The Swift helper should receive already-materialized boot artifacts.

Required phase-1 inputs:

- `kernel`
  - path to an aarch64 Linux kernel image
  - fed into `VZLinuxBootLoader(kernelURL:)`
- `initrd`
  - optional path to an initramfs image
  - fed into `initialRamdiskURL` when used
- `cmdline`
  - Linux command line for console and root-disk selection
- `root_disk`
  - writable bootable Linux root disk image
- `cloud_init_disk`
  - separate read-only NoCloud disk artifact

Phase-1 Vz storage shape should therefore be:

- one root block device that the guest actually boots from
- one independent NoCloud disk for boot-time customization

This is different from the CH examples, which currently lean on:

- direct kernel boot
- squashfs root as a read-only disk
- ext4 writable overlay
- launch-time seeding of NoCloud files into the writable layer

For Vz, cloud-init should not be hidden inside the writable root-disk mutation
path. It should be attached as its own boot artifact.

## Recommended Artifact Set

For phase 1, the Vz guest image bundle should contain:

- `Image`
  - aarch64 Linux kernel
- `initrd.img`
  - optional; only if the selected root-disk layout requires one
- `root.img`
  - writable Linux root disk image, likely raw for simplicity
- `cidata.iso` or equivalent NoCloud disk image
  - contains:
    - `user-data`
    - `meta-data`
    - optionally `network-config`

The `mounts.yaml` content used by guestfs should be delivered through
`user-data`/`write_files` or another cloud-init-managed path, not through a CH
overlay-specific mechanism.

## aarch64 Linux Requirements For Apple Vz

Phase-1 Vz should assume Apple Silicon first.

The guest kernel must satisfy:

- aarch64 Linux kernel image format suitable for `VZLinuxBootLoader`
- support for virtio block
- support for virtio console/serial
- support for virtio vsock
- support for virtio-net
- support for virtio-fs if the first Vz filesystem slice uses hypervisor-managed
  host sharing
- support for the selected root filesystem and initramfs flow

Guest userspace must also assume:

- `cloud-init` present and enabled
- `systemd`
- OpenSSH server
- `socat` for the current guest-side vsock-to-SSH bridge contract
- FUSE userspace packages if the later guestfs PoC keeps the existing
  `motlie-vfs-guest` path

`VZLinuxBootLoader` plus `VZGenericPlatformConfiguration` means phase 1 should
avoid adding EFI-specific requirements unless there is a strong reason to leave
the direct Linux boot path.

Recommended command-line defaults should stay minimal and explicit, for example:

- serial console enabled
- explicit root device
- no CH-specific `overlay-init` assumptions

## Build Tooling Options

There are several realistic ways to build the Vz guest image.

### Option A: Reuse the Current Debian Build Inputs, Change Output Packaging

Approach:

- keep Debian Bookworm/minbase style guest construction
- keep `mmdebstrap` or equivalent Linux-side rootfs build
- keep installing the same guest packages and binaries as the CH line
- convert the result into a Vz-bootable root disk plus a separate NoCloud disk

Pros:

- highest semantic reuse with the current CH/v1.4 guest contract
- easiest path to preserve:
  - cloud-init behavior
  - OpenSSH setup
  - `motlie-vfs-guest`
  - `motlie-vmm-vsock-ssh`
  - CA trust and user-provisioning assumptions

Cons:

- still Linux-host oriented if `mmdebstrap` remains the image-builder
- current CH stacked-root assumptions must be removed from the final artifact
  layout

Recommendation:

- this is the preferred long-term phase-1 direction

### Option B: `debootstrap`/`mmdebstrap` + Custom Raw-Disk Builder Script

Approach:

- build the rootfs with Debian tooling
- create a raw disk image
- partition/format it
- install the rootfs into the raw image
- separately build the NoCloud disk

Pros:

- fully explicit and repo-owned
- closest to existing CH image-builder discipline
- easiest to automate in CI

Cons:

- more custom scripting
- likely still Linux-host dependent

Recommendation:

- this is the best concrete build shape for `v1.05`

### Option C: `vftool`

`vftool` is useful as a launcher/prototyping aid, not as the primary image
pipeline.

Pros:

- helpful for local Vz experimentation on macOS

Cons:

- it does not solve the image build/distribution problem by itself
- it adds another launcher abstraction we do not want as the long-term runtime

Recommendation:

- acceptable as a debugging aid, not the owning image pipeline

### Option D: `tart`

`tart` is a polished Apple virtualization image tool, but it is aimed more at
VM image management than at preserving the exact repo-owned guest contract.

Pros:

- strong developer ergonomics on macOS
- image caching and VM management are already solved

Cons:

- can become a second image abstraction separate from the repo-owned build
  contract
- weaker fit if the goal is “same guest contract, different hypervisor
  packaging”

Recommendation:

- useful reference or dev convenience layer, but not the primary build contract

### Option E: Alpine Minimal

Pros:

- potentially much smaller image
- fast boot and build times

Cons:

- diverges sharply from the current Debian/Ubuntu-oriented guest contract
- higher parity risk for:
  - cloud-init behavior
  - package assumptions
  - service layout
  - existing scripts and debugging expectations

Recommendation:

- not recommended for phase 1 parity work

### Option F: Ubuntu Minimal

Pros:

- similar enough to the current Debian-style userspace expectations
- good cloud-init support

Cons:

- still a guest-contract fork from the current CH image line
- little benefit if the repo already knows how to build Debian-style images

Recommendation:

- acceptable fallback if Debian tooling becomes a blocker, but not the first
  choice

## Recommended Phase-1 Build Direction

Phase 1 should use:

- Debian-based rootfs build
- repo-owned custom image build script
- raw root disk output
- separate NoCloud disk creation
- guest binaries baked into the root image

That means the image pipeline should look like:

1. build guest-side Rust binaries on Linux:
   - `motlie-vfs-guest`
   - any guest helper binaries required by the current `v1.4` contract
2. assemble a Debian-based Linux rootfs with:
   - `cloud-init`
   - OpenSSH server
   - `socat`
   - FUSE packages
   - the guest helper binaries and systemd units
3. pack that rootfs into a Vz-suitable root disk image
4. render `user-data` / `meta-data`
5. build a separate NoCloud disk image

## Cloud-Init And SSH Key Injection

The Vz guest image path must preserve compatibility with the current
auto-provisioning direction in `libs/vmm`.

That means cloud-init must remain able to inject:

- the guest user account
- authorized SSH keys or principals wiring
- CA trust / sshd configuration where the current line expects it
- guest mount configuration such as `/etc/motlie-vfs/mounts.yaml`

For compatibility with the hardened SSH and auto-provision path, the Vz image
should preserve the same guest-side contract as CH:

- same SSH server behavior
- same `motlie-vmm-vsock-ssh` service or reviewed equivalent
- same user-creation flow
- same per-guest `authorized_keys` / principal expectations

Recommendation:

- deliver SSH and user customization entirely through the NoCloud disk in phase
  1
- avoid ad hoc first-boot fetchers or mutable image post-processing on macOS

This keeps the Vz image path aligned with the orchestrator-owned provisioning
model rather than moving identity concerns into an opaque VM image manager.

## Image Size Targets

Phase-1 targets should distinguish development from production.

Development target:

- root disk: roughly `1-2 GiB` compressed artifact budget, with room for
  debugging tools and package installs
- NoCloud disk: small, typically `<< 10 MiB`

Production/minimized target:

- root disk: ideally `<= 1 GiB` compressed distribution artifact
- NoCloud disk: still tiny

These are deliberately pragmatic rather than heroic.

The current CH line already values debuggability and parity over maximal size
reduction. The first Vz path should do the same.

## Distribution Packaging

The Vz guest image should be treated as a multi-artifact bundle, not as one
opaque file.

Minimum distribution contents:

- kernel
- optional initrd
- root disk
- metadata describing:
  - architecture
  - kernel version
  - rootfs distro/build
  - expected boot command line defaults

Recommended bundle shape:

- a manifest plus content-addressed artifacts
- artifact versioning tied to the guest contract revision

If an OCI/xar-style artifact format is adopted later, this Vz image bundle maps
cleanly to:

- manifest
- kernel layer/artifact
- initrd layer/artifact
- root-disk layer/artifact

Important caveat:

- there is no local `libs/xar` crate in this worktree today, so there is no
  existing packaging API to integrate directly against in phase 1

Design consequence:

- define the guest image bundle shape now
- keep the build output easy to map into a future OCI/xar packager
- do not block the image pipeline on a packaging subsystem that is not yet
  present locally

## Relationship To The Existing CH Guest Image Path

The CH and Vz guest image paths are not identical, but they are also not
fundamentally different at the guest userspace layer.

They can and should share:

- distro family choice
- package set where practical
- `cloud-init`
- OpenSSH configuration model
- guest helper binaries
- systemd units
- CA trust and user-provisioning semantics

They should not be forced to share:

- hypervisor-specific boot artifact layout
- kernel packaging assumptions
- root-disk wiring
- launch-time NoCloud delivery mechanism
- CH stacked-root `overlay-init` assumptions

So the right mental model is:

- shared guest contract
- different hypervisor realization

That is the key distinction needed before `#172`.

## Phase-1 Recommendation

Before `#172`, build and validate a narrow `v1.05` image PoC that proves:

- an aarch64 Linux guest boots under Vz
- NoCloud delivery works
- `motlie-vfs-guest` and the boot-time guest services are present
- the guest userspace contract is close enough to CH that later guestfs and SSH
  work can build on it

Only after that should the Vz guestfs PoC claim that guestfs failures are about
transport or semantics rather than about a missing or unstable guest image
contract.
