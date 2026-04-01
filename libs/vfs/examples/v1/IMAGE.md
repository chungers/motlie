# Guest Image Builder

`build-guest.sh` creates the squashfs root image and ext4 overlay for
Cloud Hypervisor guests. **No sudo required** — the script uses
`mmdebstrap --mode=unshare` for fully rootless image construction.

## Prerequisites

```bash
sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring
```

On Ubuntu 24.04, unprivileged user namespaces must be allowed:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

To make it persistent:

```bash
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/99-userns.conf
```

You also need `/etc/subuid` and `/etc/subgid` entries for your user
(typically set up by default on Ubuntu):

```bash
grep $USER /etc/subuid /etc/subgid
# Should show something like: dchung:100000:65536
```

## Usage

```bash
# Full build: guest binary + kernel download + image
./build-guest.sh

# With pre-built binary, skip kernel (fastest rebuild)
./build-guest.sh --guest-binary ../../../../target/release/motlie-vfs-guest --kernel skip

# Kernel modes
./build-guest.sh --kernel download   # (default) pre-built from cloud-hypervisor/linux
./build-guest.sh --kernel build      # build from source (slow, needs build-essential)
./build-guest.sh --kernel skip       # use existing kernel in artifacts/
```

## What it produces

```
artifacts/
├── Image              # CH-compatible aarch64 kernel (or vmlinux.bin on x86_64)
├── rootfs.squashfs    # Debian Bookworm root (~60-80 MB)
└── overlay.ext4       # Writable ext4 overlay (64 MB, sparse)
```

## How it works

### Rootless image building with mmdebstrap

Traditional `debootstrap` requires root for `chroot`, device node creation,
and file ownership. `mmdebstrap --mode=unshare` avoids this by using Linux
user namespaces:

1. Creates a user namespace where the current user maps to uid 0
2. Inside the namespace, runs `debootstrap` with full root privileges
3. File ownership (uid 0 for root, uid 1000 for alice) is mapped correctly
4. Outputs directly to squashfs format — no intermediate rootfs directory

The `uidmap` package provides `newuidmap`/`newgidmap` which handle the
uid/gid mapping between the namespace and the host.

### Build steps

| Step | What | Rootless? |
|------|------|-----------|
| 1. Guest binary | `cargo build --release --features vsock,client` | Yes |
| 2. Kernel | Download pre-built or build from source | Yes |
| 3. Squashfs root | `mmdebstrap --mode=unshare --format=squashfs` | Yes |
| 4. ext4 overlay | `mkfs.ext4 -d` (populate from directory) | Yes |

### Squashfs contents

The rootfs includes:

- **Base**: Debian Bookworm minbase (systemd, bash, coreutils)
- **Network**: openssh-server, iproute2
- **FUSE**: fuse3, libfuse3-3, user_allow_other in fuse.conf
- **Tools**: tmux
- **Guest agent**: `/usr/local/bin/motlie-vfs-guest` (systemd service)
- **Boot**: `/sbin/overlay-init` (squashfs + ext4 → overlayfs → pivot_root)
- **User**: alice (uid=1000, password `testpass`), root (password `rootpass`)
- **Login UX**: MOTD art, tmux auto-prompt, ~/.env auto-source

### ext4 overlay contents

Pre-seeded with:

```
upper/
├── etc/motlie-vfs/
│   └── mounts.yaml        # Guest mount config (tag: alice-home → /home/alice)
└── home/alice/
    ├── .ssh/               # Mode 700, mount point for FUSE overlay
    └── workspace/          # Empty workspace directory
```

### Customization

To add packages, add them to the `--include=` list in `build-guest.sh`.

To run additional setup commands, add `--customize-hook` entries. Each
hook receives the rootfs path as `$1`:

```bash
--customize-hook='chroot "$1" apt-get install -y vim'
--customize-hook='echo "export EDITOR=vim" >> "$1/etc/profile.d/editor.sh"'
```

## Comparison with previous approach

| | Old (debootstrap) | New (mmdebstrap) |
|---|---|---|
| Root required | Yes (`sudo`) | No |
| Mechanism | chroot + mksquashfs | user namespaces |
| Extra packages | uidmap not needed | `uidmap` required |
| Output format | Intermediate dir → squashfs | Direct squashfs |
| Artifact ownership | Root-owned, needs chown | User-owned |
| Speed | Similar | Similar |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `mmdebstrap: command not found` | `sudo apt install mmdebstrap` |
| `need tar2sqfs binary` | `sudo apt install squashfs-tools-ng` |
| `newuidmap: command not found` | `sudo apt install uidmap` |
| `unshare: Operation not permitted` | `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` |
| `No matching subuids/subgids` | Check `/etc/subuid` and `/etc/subgid` have entries for your user |
| `tar2sqfs does not support extended attributes` | Harmless — ACL/security xattrs from systemd packages. squashfs works fine without them |
| `failed to open configuration file .dpkg.cfg` | Harmless — uid remapping inside user namespace prevents reading host home dir. No dpkg config override exists anyway |
| Squashfs too large | Add `--customize-hook='rm -rf "$1/usr/share/doc"'` to strip docs |
