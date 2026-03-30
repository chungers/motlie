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
- Alpine Linux minimal (sshd, bash, coreutils)
- `motlie-vfs-guest` binary (statically linked, musl)
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
sudo apt install squashfs-tools e2fsprogs wget    # Debian/Ubuntu
# or
sudo apk add squashfs-tools e2fsprogs wget        # Alpine

# Cloud Hypervisor
wget https://github.com/cloud-hypervisor/cloud-hypervisor/releases/download/v44.0/cloud-hypervisor-static
chmod +x cloud-hypervisor-static
sudo mv cloud-hypervisor-static /usr/local/bin/cloud-hypervisor

# vsock kernel module
sudo modprobe vhost_vsock
```

### Cross-compiling from macOS

The guest binary must be a statically-linked x86_64 Linux ELF. From macOS:

```bash
# Option A: Using cross (recommended — Docker-based, zero setup)
cargo install cross --git https://github.com/cross-rs/cross
cross build --release --target x86_64-unknown-linux-musl -p motlie-vfs --bin motlie-vfs-guest

# Option B: Using homebrew musl-cross
brew install filosottile/musl-cross/musl-cross
rustup target add x86_64-unknown-linux-musl
CC_x86_64_unknown_linux_musl="x86_64-linux-musl-gcc" \
    cargo build --release --target x86_64-unknown-linux-musl -p motlie-vfs --bin motlie-vfs-guest
```

The image build itself (`build-guest.sh`) must run on Linux because it uses
`chroot` and `mksquashfs`. Run it in a Linux VM, CI, or Docker container.

## Step-by-Step

### 1. Build the kernel

Cloud Hypervisor needs a kernel with squashfs, overlayfs, and vsock support.
The CH project maintains a kernel branch with their recommended config:

```bash
git clone --depth 1 https://github.com/cloud-hypervisor/linux.git \
    -b ch-6.12.8 linux-cloud-hypervisor
cd linux-cloud-hypervisor

# Start from CH's default config
make ch_defconfig

# Ensure squashfs, overlayfs, and vsock are enabled
cat >> .config << 'EOF'
CONFIG_SQUASHFS=y
CONFIG_SQUASHFS_ZSTD=y
CONFIG_OVERLAY_FS=y
EOF
make olddefconfig

# Build
make bzImage -j$(nproc)

# Copy the kernel
cp arch/x86/boot/compressed/vmlinux.bin /path/to/libs/vfs/image/artifacts/vmlinux.bin
```

### 2. Cross-compile the guest binary

```bash
# From the workspace root, on macOS or Linux
cross build --release --target x86_64-unknown-linux-musl -p motlie-vfs --bin motlie-vfs-guest

# Verify it's a static musl binary
file target/x86_64-unknown-linux-musl/release/motlie-vfs-guest
# → ELF 64-bit LSB executable, x86-64, statically linked
```

### 3. Build the guest images

```bash
cd libs/vfs/image

# Pass the cross-compiled binary explicitly, or let the script find it
./build-guest.sh --guest-binary ../../target/x86_64-unknown-linux-musl/release/motlie-vfs-guest

# Output:
#   artifacts/rootfs.squashfs  (~30-50 MB)
#   artifacts/overlay.ext4     (64 MB, sparse)
```

### 4. Launch the guest

```bash
./launch-ch.sh
```

The guest boots to a serial console. You'll see Alpine's OpenRC init sequence
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

In a separate terminal on the host:

```bash
# The host FsServer will listen on the vsock socket.
# When the guest's motlie-vfs-guest connects to vsock CID 2 port 5000,
# it appears as a Unix connection on /tmp/motlie-vfs.vsock_5000.
#
# For now, use the example host binary or a custom Rust program:
cargo run -p motlie-vfs --example simple_host --features vsock
```

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
| CH fails with permission error | Run with `sudo` or grant `CAP_NET_ADMIN` |
| Guest kernel panic at boot | Ensure kernel has `CONFIG_SQUASHFS=y CONFIG_OVERLAY_FS=y` |
| No serial output | Check `--serial tty --console off` in CH args |
| SSH connection refused | Wait for sshd to start (~2s after boot), check `192.168.249.2` |
| vsock socket not created | Check guest connects to CID 2, correct port |
