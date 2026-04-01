# v1.1 Guest Image Notes

`build-guest.sh` in `v1.1` builds one generic shared base image set.

## Original Requirements

`v1.1` was changed to satisfy these requirements:

- build one generic image set, not separate Alice/Bob images
- keep guest identity out of the build artifact
- let Alice install software such as `python3` without Bob seeing it
- discard those guest-local installs on the next guest launch
- keep `vfs` subtree mounts such as `/home/alice`, `/home/bob`, and `/workspace` working on top of the guest root

Those requirements define the artifact boundary for `v1.1`.

## Artifact Contract

`v1.1` has exactly two artifact classes:

1. Build-time shared base artifacts
2. Launch-time per-guest writable artifacts

There is no third deployed image layer. `overlay.d/` is only a source tree
used to seed the launch-time writable overlay.

### Build-Time Shared Base

The shared base is immutable and common to every guest:

- `rootfs.squashfs`
- `Image` or `vmlinux.bin`
- package-managed software
- common tools and shared libraries
- common `/usr/local/bin` content that every guest should always see

### Launch-Time Per-Guest Writable Overlay

The launch-time overlay is guest-local and disposable:

- one fresh ext4 image per guest launch
- mounted as the writable upper layer for `/`
- holds guest identity such as `mounts.yaml` and hostname
- holds guest-local mutations such as `apt install python3`
- recreated on the next launch, so guest-local installs do not persist

That means Alice and Bob never share a writable root overlay, and Alice's
package installs do not leak to Bob or to Alice's next boot.

## Output

The builder writes:

- `artifacts/base/rootfs.squashfs`
- `artifacts/base/Image` or `artifacts/base/vmlinux.bin`

Example:

```text
artifacts/
  base/
    Image
    rootfs.squashfs
```

## Usage

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh
./build-guest.sh --guest-binary /path/to/motlie-vfs-guest
./build-guest.sh --kernel skip
```

## What The Builder Does

`build-guest.sh` performs three steps:

1. Builds the shared `motlie-vfs-guest` binary from the workspace root unless `--guest-binary` is supplied.
2. Downloads or builds a shared Cloud Hypervisor-compatible kernel unless `--kernel skip` is used.
3. Builds a shared Debian Bookworm squashfs root image with:
   - `openssh-server`
   - `bash`, `coreutils`, `tmux`
   - `fuse3`, `libfuse3-3`
   - `systemd`, `dbus`, `iproute2`
   - both demo users: `alice` and `bob`
   - `motlie-vfs-guest`
   - `overlay-init`
Guest-specific writable overlays are not built here. `launch-ch.sh` creates
them at boot time from guest-specific runtime seed content.

## Runtime Guest State

At launch time, `launch-ch.sh` creates one per-guest writable ext4 overlay
under `${RUNTIME_ROOT:-/tmp/motlie-vfs-v11-runtime}/<guest>/overlay.ext4`.

That runtime overlay seeds:

- `/etc/motlie-vfs/mounts.yaml`
- `/etc/hostname`
- `/home/<guest>/.ssh`
- `/workspace`
- optional files from `overlay.d/common` and `overlay.d/<guest>`

This ext4 image is the writable upper layer for the guest root. It starts
small and sparse, but it can absorb writes anywhere under `/`, including
package-manager state under `/usr`, `/etc`, and `/var`.

## Prerequisites

```bash
sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring
sudo apt install libfuse3-dev pkg-config
```

Also required:

- outbound network access for Debian packages and kernel download
- a login shell whose primary gid matches the passwd entry for the user when using `mmdebstrap --mode=unshare`
- `mkfs.ext4` at launch time for runtime overlay creation (`sudo apt install e2fsprogs`)

On Ubuntu 24.04:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

## Build Variants

Use a prebuilt guest binary:

```bash
./build-guest.sh --guest-binary /path/to/motlie-vfs-guest
```

Reuse an already downloaded or prepositioned kernel:

```bash
./build-guest.sh --kernel skip
```

Build the kernel from source instead of downloading it:

```bash
./build-guest.sh --kernel build
```

Use a rootful fallback when `--mode=unshare` is blocked:

```bash
MMDEBSTRAP_MODE=root ./build-guest.sh
```

## What Differs Vs v1

- `v1` builds one artifact set in `artifacts/`; `v1.1` builds one generic shared base in `artifacts/base/`
- `v1` seeds one mount config file during the build; `v1.1` injects `mounts.alice.yaml` or `mounts.bob.yaml` only at launch
- `v1` is centered on one guest user (`alice`); `v1.1` includes both `alice` and `bob` in the shared base and specializes only the launch-time overlay per guest
- `v1` demonstrates one guest VM and typically one mount tag; `v1.1` is built to support multiple guests and multiple tags per guest
- `v1.1` creates per-guest runtime writable overlays at launch time instead of baking guest identity during the build
- `v1.1` supports optional overlay content injection from `overlay.d/` so guest-local changes do not force a squashfs rebuild

## Where To Put Changes

Put changes in the shared base when they are:

- common to every guest
- coarse-grained
- package-managed
- dependent on shared libraries or system integration

Put changes in the launch-time overlay when they are:

- guest-specific
- experimental
- disposable
- configuration or identity data
- self-contained scripts or binaries you want only one guest to see

Examples:

- install `python3` for every guest: shared base
- let Alice try `apt install python3` for one run: launch-time overlay
- add a common helper in `/usr/local/bin` for every guest: shared base
- add `overlay.d/alice/usr/local/bin/demo-tool` for Alice only: launch-time overlay

## Overlay Content

`overlay.d/common/` and `overlay.d/<guest>/` are copied into the runtime
overlay `upper/` tree during `launch-ch.sh`.

That means you can add content like:

```text
overlay.d/common/usr/local/bin/demo-tool
overlay.d/alice/etc/profile.d/alice-demo.sh
```

This is a good fit for:

- shell scripts
- small helper binaries
- self-contained tools placed under `/usr/local/bin`
- guest-local experiments you want discarded on the next launch

This is not the best fit for:

- package-managed software
- binaries that need shared libraries not already in the base rootfs
- large additions that should live in the shared base image instead

## Standalone Expectation

You do not need to build `examples/v1` first. `examples/v1.1` contains its own:

- `build-guest.sh`
- `launch-ch.sh`
- `overlay-init`
- `overlay.d/`
- mount configs
- demo REPL scripts

The only shared dependency is the workspace binary build of `motlie-vfs-guest`, which is expected because `v1.1` is an example harness inside the same repository.
