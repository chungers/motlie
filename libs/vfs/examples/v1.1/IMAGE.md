# v1.1 Guest Image Notes

`build-guest.sh` in `v1.1` builds guest-specific artifacts under `artifacts/<guest>/`.

## Usage

```bash
./build-guest.sh --guest alice
./build-guest.sh --guest bob
./build-guest.sh --guest alice --guest-binary /path/to/motlie-vfs-guest
./build-guest.sh --guest bob --kernel skip
```

## What Changes Relative To v1

- The rootfs contains both demo users:
  - `alice` uid/gid `1000:1000`
  - `bob` uid/gid `1001:1001`
- The overlay image is guest-specific and seeds the selected mount config as `/etc/motlie-vfs/mounts.yaml`.
- Artifacts are split by guest name:
  - `artifacts/alice/`
  - `artifacts/bob/`

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

On Ubuntu 24.04:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

See `../v1/IMAGE.md` for the underlying rootless `mmdebstrap` rationale. `v1.1` reuses the same build mechanism and only changes guest-specific parameterization.
