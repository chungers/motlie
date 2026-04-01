# motlie-vfs Guest Image Builder

Build and launch a minimal Linux guest in Cloud Hypervisor for testing the
motlie-vfs vsock + overlay stack end-to-end.

## Architecture

```
Host (macOS or Linux)                     Guest (Cloud Hypervisor VM)
─────────────────────                    ──────────────────────────
FsServer + MemOverlay                    motlie-vfs-guest
  ↕ vsock (port 5000)                      ↕ FUSE mount
/tmp/motlie-vfs.vsock_5000               /home/alice (overlay + disk)
```

**Guest root model:** read-only squashfs base + writable ext4 overlay, stacked
by `overlay-init` at boot via overlayfs. This matches the `motlie-vmm` design.

**Guest image contents:**
- Debian Bookworm minimal (sshd, bash, coreutils, tmux, fuse3, systemd)
- `motlie-vfs-guest` binary (dynamically linked, GNU libc)
- `overlay-init` script (squashfs + ext4 → overlayfs → pivot_root)
- `/etc/motlie-vfs/mounts.yaml` (seeded in the ext4 overlay)
- Test user `alice` (uid=1000, gid=1000, password `testpass`)

## Files

| File | Purpose |
|------|---------|
| `build-guest.sh` | Builds squashfs + ext4 images |
| `launch-ch.sh` | Launches Cloud Hypervisor with vsock + networking |
| `overlay-init` | Boot-time init script baked into squashfs |
| `mounts.yaml` | Default mount config seeded into ext4 overlay |
| `artifacts/` | Build output (gitignored) |

## Prerequisites

### On Linux (build host)

```bash
# Package dependencies
# Image builder (rootless — no sudo needed for build-guest.sh)
sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring libfuse3-dev pkg-config
# On Ubuntu 24.04, also allow unprivileged user namespaces:
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

# Cloud Hypervisor — download the binary for your architecture
# x86_64:
wget https://github.com/cloud-hypervisor/cloud-hypervisor/releases/download/v44.0/cloud-hypervisor-static
chmod +x cloud-hypervisor-static
sudo mv cloud-hypervisor-static /usr/local/bin/cloud-hypervisor

# aarch64:
wget https://github.com/cloud-hypervisor/cloud-hypervisor/releases/download/v44.0/cloud-hypervisor-static-aarch64
chmod +x cloud-hypervisor-static-aarch64
sudo mv cloud-hypervisor-static-aarch64 /usr/local/bin/cloud-hypervisor

# vsock kernel module
sudo modprobe vhost_vsock
```

### Building the guest binary

On a native Linux host with `libfuse3-dev` installed, `build-guest.sh`
compiles the guest binary automatically using the default cargo target
(dynamically linked GNU libc). The Debian-based guest image includes the
required shared libraries (`libfuse3`, `libc`, etc.).

```bash
# Automatic (build-guest.sh handles this):
cargo build --release --features vsock,client -p motlie-vfs --bin motlie-vfs-guest

# Or supply a pre-built binary:
./build-guest.sh --guest-binary /path/to/motlie-vfs-guest
```

### Cross-compiling from macOS

From macOS, use `cross` (Docker-based) to produce a Linux binary:

```bash
cargo install cross --git https://github.com/cross-rs/cross

# x86_64 guest:
cross build --release --target x86_64-unknown-linux-gnu --features vsock,client -p motlie-vfs --bin motlie-vfs-guest

# aarch64 guest:
cross build --release --target aarch64-unknown-linux-gnu --features vsock,client -p motlie-vfs --bin motlie-vfs-guest
```

The image build itself (`build-guest.sh`) must run on Linux because it uses
`mmdebstrap --mode=unshare` (Linux user namespaces). No sudo required.
Run it on a Linux host, in CI, or in a Docker container with user namespace
support. See [IMAGE.md](IMAGE.md) for full prerequisites.

## Step-by-Step

### 1. Build everything

`build-guest.sh` handles kernel, guest binary, and images in one command:

```bash
cd libs/vfs/examples/v1

# Default: downloads pre-built kernel, builds guest binary, creates images
# No sudo required — uses mmdebstrap for rootless image building.
./build-guest.sh

# With a pre-built guest binary:
./build-guest.sh --guest-binary /path/to/motlie-vfs-guest

# Kernel modes:
./build-guest.sh --kernel download   # (default) pre-built from cloud-hypervisor/linux
./build-guest.sh --kernel build      # clone and build from source
./build-guest.sh --kernel skip       # use existing kernel in artifacts/
```

Output in `artifacts/`:
- `Image` or `vmlinux.bin` — CH-compatible kernel
- `rootfs.squashfs` — Debian root image (~80-120 MB)
- `overlay.ext4` — writable ext4 overlay (64 MB, sparse)

The pre-built kernel comes from the
[cloud-hypervisor/linux releases](https://github.com/cloud-hypervisor/linux/releases)
(`ch-release-v6.16.9-20251112`). It includes squashfs, overlayfs, and
virtio-vsock support — the same kernel CH's own CI tests against.

To build from source instead (requires git, make, gcc, flex, bison,
libelf-dev, libssl-dev):

```bash
./build-guest.sh --kernel build
```

### 2. Launch the guest

```bash
./launch-ch.sh
```

The guest boots to a serial console. You'll see Debian's systemd init sequence
and a login prompt. The boot takes ~1-2 seconds.

### 5. Validate

**SSH access (if TAP networking is enabled):**

```bash
ssh alice@192.168.249.2    # password: testpass
```

**Inside the guest:**

```bash
# Verify the overlay mount
mount | grep overlay
# → overlay on / type overlay (rw,lowerdir=/mnt/lower,upperdir=...)

# Verify motlie-vfs-guest is installed
which motlie-vfs-guest
motlie-vfs-guest --help

# Check the mount config
cat /etc/motlie-vfs/mounts.yaml

# Verify alice user
id alice
# → uid=1000(alice) gid=1000(alice)
```

### 6. Start the host-side server

**Start the host server before booting the guest** (or in a separate terminal):

```bash
# Starts one FsServer per guest VM. Each VM gets its own socket.
# Default: socket /tmp/motlie-vfs.vsock_5000, tag alice-home, temp host dir.
cargo run -p motlie-vfs --example repl_host --features vsock

# With explicit parameters (for multi-guest, run separate instances):
cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --socket /tmp/motlie-vfs-alice.vsock_5000 \
    --tag alice-home \
    --dir /path/to/alice/home

# Second guest VM in another terminal:
cargo run -p motlie-vfs --example repl_host --features vsock -- \
    --socket /tmp/motlie-vfs-bob.vsock_5000 \
    --tag bob-home \
    --dir /path/to/bob/home
```

The host server provides a rustyline REPL for overlay mutation. Example
SSH key injection workflow for the `alice-home` tag (uid=1000 gid=1000):

```
vfs> layer credentials 0
ok: layer credentials priority=0

vfs> putattr credentials alice-home /.ssh/authorized_keys 1000 1000 600 ssh-ed25519 AAAA... alice@dev
ok: putattr credentials alice-home /.ssh/authorized_keys uid=1000 gid=1000 mode=600 (38 bytes)

vfs> putattr credentials alice-home /.ssh/config 1000 1000 644 Host github.com
ok: putattr credentials alice-home /.ssh/config uid=1000 gid=1000 mode=644 (19 bytes)

vfs> put credentials alice-home /.env ANTHROPIC_API_KEY=sk-ant-xxx
ok: put credentials alice-home /.env (31 bytes)

vfs> ls alice-home
  Content { size: 38 } /.ssh/authorized_keys uid=1000 gid=1000 mode=600
  Content { size: 19 } /.ssh/config uid=1000 gid=1000 mode=644
  Content { size: 31 } /.env uid=0 gid=0 mode=644
(3 entries)
```

Type `help` for all available commands.

### 7. Shut down the guest

```bash
# From the host
curl --unix-socket /tmp/motlie-vfs-api.sock -X PUT http://localhost/api/v1/vm.shutdown

# Or from inside the guest
poweroff
```

## vsock Communication Model

Cloud Hypervisor uses the Firecracker vsock model:

| Direction | Host side | Guest side |
|-----------|-----------|------------|
| Guest → Host | Unix socket at `<socket>_<port>` | `connect(CID=2, port)` |
| Host → Guest | `echo "CONNECT <port>" \| socat - UNIX:<socket>` | `listen(port)` |

For motlie-vfs:
- Host `FsServer` listens on `/tmp/motlie-vfs.vsock_5000`
- Guest `motlie-vfs-guest` connects to vsock CID 2, port 5000
- The `VsockConnectionHandler` serves the connection

## Modifying the Config

Edit `mounts.yaml` before running `build-guest.sh` to change the mount layout:

```yaml
mounts:
  - tag: alice-home
    guest_path: /home/alice
    read_only: false
  - tag: workspace
    guest_path: /workspace
    read_only: false
```

To update config on an existing ext4 overlay without rebuilding:

```bash
# Mount the overlay image, update the config, unmount
mkdir -p /tmp/overlay-mnt
sudo mount -o loop artifacts/overlay.ext4 /tmp/overlay-mnt
sudo cp mounts.yaml /tmp/overlay-mnt/upper/etc/motlie-vfs/mounts.yaml
sudo umount /tmp/overlay-mnt
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `vhost_vsock` not found | `sudo modprobe vhost_vsock` |
| `mmdebstrap: command not found` | `sudo apt install mmdebstrap squashfs-tools-ng uidmap debian-archive-keyring` |
| `unshare: Operation not permitted` | `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` |
| CH disk `PermissionDenied` | Artifacts should be user-owned (rootless build). If stale root-owned artifacts remain from an old sudo build: `sudo chown $USER:$USER artifacts/*` |
| CH TAP `Operation not permitted` | TAP networking needs `CAP_NET_ADMIN`. Use `./launch-ch.sh --no-net` for vsock-only, or grant the capability: `sudo setcap cap_net_admin+ep $(which cloud-hypervisor)` |
| CH fails with KVM permission error | Ensure user is in `kvm` group (`sudo usermod -aG kvm $USER`, then re-login) |
| Guest kernel panic at boot | Ensure kernel has `CONFIG_SQUASHFS=y CONFIG_OVERLAY_FS=y` |
| No serial output | Check `--serial tty --console off` in CH args |
| SSH connection refused | Wait for sshd to start (~2s after boot), check `192.168.249.2` |
| vsock socket not created | Check guest connects to CID 2, correct port |
