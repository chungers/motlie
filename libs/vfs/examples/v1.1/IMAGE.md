# v1.1 Guest Image Notes

`build-guest.sh` in `v1.1` builds guest-specific artifacts under `artifacts/<guest>/`.

## Output

For each guest, the builder writes:

- `artifacts/<guest>/rootfs.squashfs`
- `artifacts/<guest>/overlay.ext4`
- `artifacts/<guest>/Image` or `artifacts/<guest>/vmlinux.bin`

Example:

```text
artifacts/
  alice/
    rootfs.squashfs
    overlay.ext4
    Image
  bob/
    rootfs.squashfs
    overlay.ext4
    Image
```

## Usage

```bash
cd libs/vfs/examples/v1.1

./build-guest.sh --guest alice
./build-guest.sh --guest bob
./build-guest.sh --guest alice --guest-binary /path/to/motlie-vfs-guest
./build-guest.sh --guest bob --kernel skip
```

## What The Builder Does

`build-guest.sh` performs four steps for the selected guest:

1. Builds the shared `motlie-vfs-guest` binary from the workspace root unless `--guest-binary` is supplied.
2. Downloads or builds a Cloud Hypervisor-compatible kernel unless `--kernel skip` is used.
3. Builds a Debian Bookworm squashfs root image with:
   - `openssh-server`
   - `bash`, `coreutils`, `tmux`
   - `fuse3`, `libfuse3-3`
   - `systemd`, `dbus`, `iproute2`
   - both demo users: `alice` and `bob`
   - `motlie-vfs-guest`
   - `overlay-init`
4. Builds a guest-specific ext4 overlay image that seeds:
   - `/etc/motlie-vfs/mounts.yaml`
   - `/home/<guest>/.ssh`
   - `/workspace`

## Guest-Specific Inputs

The selected guest controls:

- which mount config is injected:
  - `mounts.alice.yaml`
  - `mounts.bob.yaml`
- which login home is pre-created in the overlay:
  - `/home/alice`
  - `/home/bob`
- which artifact directory is written:
  - `artifacts/alice/`
  - `artifacts/bob/`
- which hostname is written into the rootfs:
  - `motlie-alice`
  - `motlie-bob`

## Seeded Mount Points

For the selected guest, the overlay image seeds:

- `/home/<guest>/.ssh`
- `/workspace`

Those directories must exist before `motlie-vfs-guest` mounts the FUSE filesystems there.

## Prerequisites

```bash
sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring
sudo apt install libfuse3-dev pkg-config
```

Also required:

- outbound network access for Debian packages and kernel download
- a login shell whose primary gid matches the passwd entry for the user when using `mmdebstrap --mode=unshare`

On Ubuntu 24.04:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

## Build Variants

Use a prebuilt guest binary:

```bash
./build-guest.sh --guest alice --guest-binary /path/to/motlie-vfs-guest
```

Reuse an already downloaded or prepositioned kernel:

```bash
./build-guest.sh --guest bob --kernel skip
```

Build the kernel from source instead of downloading it:

```bash
./build-guest.sh --guest alice --kernel build
```

## What Differs Vs v1

- `v1` builds one artifact set in `artifacts/`; `v1.1` builds one artifact set per guest in `artifacts/<guest>/`
- `v1` seeds one mount config file; `v1.1` selects between `mounts.alice.yaml` and `mounts.bob.yaml`
- `v1` is centered on one guest user (`alice`); `v1.1` includes both `alice` and `bob` in the rootfs and then specializes the overlay per guest
- `v1` demonstrates one guest VM and typically one mount tag; `v1.1` is built to support multiple guests and multiple tags per guest

## Standalone Expectation

You do not need to build `examples/v1` first. `examples/v1.1` contains its own:

- `build-guest.sh`
- `launch-ch.sh`
- `overlay-init`
- mount configs
- demo REPL scripts

The only shared dependency is the workspace binary build of `motlie-vfs-guest`, which is expected because `v1.1` is an example harness inside the same repository.
