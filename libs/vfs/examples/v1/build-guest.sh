#!/usr/bin/env bash
# build-guest.sh — Build the squashfs + ext4 guest image for Cloud Hypervisor.
#
# Produces:
#   artifacts/rootfs.squashfs  — read-only Debian root with sshd, bash, fuse3, motlie-vfs-guest
#   artifacts/overlay.ext4     — writable ext4 overlay pre-seeded with mounts.yaml
#   artifacts/Image|vmlinux.bin — CH-compatible kernel
#
# Prerequisites (Linux host):
#   - squashfs-tools (mksquashfs)
#   - e2fsprogs (mkfs.ext4)
#   - debootstrap
#   - sudo (for chroot-based rootfs assembly)
#   - For --kernel build: git, make, gcc, flex, bison, libelf-dev, libssl-dev
#
# Usage:
#   ./build-guest.sh [--guest-binary /path/to/motlie-vfs-guest] [--kernel download|build|skip]
#
# --kernel modes:
#   download  — (default) download pre-built kernel from cloud-hypervisor/linux releases
#   build     — clone cloud-hypervisor/linux and build from source
#   skip      — assume kernel already exists in artifacts/
#
# If --guest-binary is not provided, the script builds it with the native
# cargo toolchain (dynamically linked, requires libfuse3-dev on the host).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS="$SCRIPT_DIR/artifacts"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
GUEST_BINARY=""
KERNEL_MODE="download"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest-binary) GUEST_BINARY="$2"; shift 2 ;;
        --kernel) KERNEL_MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$KERNEL_MODE" != "download" && "$KERNEL_MODE" != "build" && "$KERNEL_MODE" != "skip" ]]; then
    echo "ERROR: --kernel must be one of: download, build, skip"
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect host architecture
# ---------------------------------------------------------------------------
HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64)
        RUST_TARGET="x86_64-unknown-linux-gnu"
        DEBOOTSTRAP_ARCH="amd64"
        KERNEL_IMAGE="vmlinux.bin"
        KERNEL_RELEASE_ASSET="vmlinux"
        KERNEL_BUILD_TARGET="bzImage"
        KERNEL_BUILD_OUTPUT="arch/x86/boot/compressed/vmlinux.bin"
        ;;
    aarch64)
        RUST_TARGET="aarch64-unknown-linux-gnu"
        DEBOOTSTRAP_ARCH="arm64"
        KERNEL_IMAGE="Image"
        KERNEL_RELEASE_ASSET="Image-arm64"
        KERNEL_BUILD_TARGET="Image"
        KERNEL_BUILD_OUTPUT="arch/arm64/boot/Image"
        ;;
    *)
        echo "ERROR: unsupported host architecture: $HOST_ARCH"
        exit 1
        ;;
esac

# Pre-built kernel release from cloud-hypervisor/linux
CH_KERNEL_RELEASE="ch-release-v6.16.9-20251112"
CH_KERNEL_URL="https://github.com/cloud-hypervisor/linux/releases/download/${CH_KERNEL_RELEASE}/${KERNEL_RELEASE_ASSET}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBIAN_SUITE="bookworm"
DEBIAN_MIRROR="http://deb.debian.org/debian"
ROOTFS_DIR="/tmp/motlie-vfs-rootfs-$$"
OVERLAY_SEED="/tmp/motlie-vfs-overlay-seed-$$"
OVERLAY_SIZE="64M"

echo "=== motlie-vfs guest image builder ==="
echo "Host arch:        $HOST_ARCH"
echo "Rust target:      $RUST_TARGET"
echo "Debootstrap arch: $DEBOOTSTRAP_ARCH"
echo "Debian suite:     $DEBIAN_SUITE"
echo "Kernel mode:      $KERNEL_MODE"
echo "Artifacts dir:    $ARTIFACTS"
echo "Workspace root:   $WORKSPACE_ROOT"

mkdir -p "$ARTIFACTS"

# ---------------------------------------------------------------------------
# Step 1: Build the guest binary (if not provided)
# ---------------------------------------------------------------------------
if [ -z "$GUEST_BINARY" ]; then
    echo ""
    echo "--- Step 1: Building motlie-vfs-guest ($RUST_TARGET) ---"

    # Native build with GNU target (dynamically linked).
    # The guest image will include the required shared libraries.
    (cd "$WORKSPACE_ROOT" && cargo build --release \
        --features vsock,client \
        -p motlie-vfs --bin motlie-vfs-guest)
    GUEST_BINARY="$WORKSPACE_ROOT/target/release/motlie-vfs-guest"

    echo "Guest binary: $GUEST_BINARY"
    file "$GUEST_BINARY"
fi

if [ ! -f "$GUEST_BINARY" ]; then
    echo "ERROR: guest binary not found at $GUEST_BINARY"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Obtain the CH-compatible kernel
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Kernel ($KERNEL_MODE) ---"

case "$KERNEL_MODE" in
    download)
        if [ -f "$ARTIFACTS/$KERNEL_IMAGE" ]; then
            echo "Kernel already exists: $ARTIFACTS/$KERNEL_IMAGE ($(du -h "$ARTIFACTS/$KERNEL_IMAGE" | cut -f1))"
        else
            echo "Downloading pre-built kernel from cloud-hypervisor/linux..."
            echo "  Release: $CH_KERNEL_RELEASE"
            echo "  URL:     $CH_KERNEL_URL"
            wget -q --show-progress -O "$ARTIFACTS/$KERNEL_IMAGE" "$CH_KERNEL_URL"
            echo "Kernel: $ARTIFACTS/$KERNEL_IMAGE ($(du -h "$ARTIFACTS/$KERNEL_IMAGE" | cut -f1))"
        fi
        ;;
    build)
        if [ -f "$ARTIFACTS/$KERNEL_IMAGE" ]; then
            echo "Kernel already exists: $ARTIFACTS/$KERNEL_IMAGE — rebuilding..."
        fi
        KERNEL_SRC="/tmp/motlie-vfs-kernel-$$"
        echo "Cloning cloud-hypervisor/linux into $KERNEL_SRC..."
        git clone --depth 1 https://github.com/cloud-hypervisor/linux.git \
            -b ch-6.12.8 "$KERNEL_SRC"
        (
            cd "$KERNEL_SRC"
            make ch_defconfig
            cat >> .config << 'KEOF'
CONFIG_SQUASHFS=y
CONFIG_SQUASHFS_ZSTD=y
CONFIG_OVERLAY_FS=y
KEOF
            make olddefconfig
            make "$KERNEL_BUILD_TARGET" -j"$(nproc)"
            cp "$KERNEL_BUILD_OUTPUT" "$ARTIFACTS/$KERNEL_IMAGE"
        )
        rm -rf "$KERNEL_SRC"
        echo "Kernel: $ARTIFACTS/$KERNEL_IMAGE ($(du -h "$ARTIFACTS/$KERNEL_IMAGE" | cut -f1))"
        ;;
    skip)
        if [ ! -f "$ARTIFACTS/$KERNEL_IMAGE" ]; then
            echo "WARNING: kernel not found at $ARTIFACTS/$KERNEL_IMAGE"
            echo "Place a CH-compatible kernel there before running launch-ch.sh."
        else
            echo "Using existing kernel: $ARTIFACTS/$KERNEL_IMAGE ($(du -h "$ARTIFACTS/$KERNEL_IMAGE" | cut -f1))"
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# Step 3: Build the Debian squashfs root image
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Building Debian ($DEBIAN_SUITE) squashfs root image ---"

echo "Bootstrapping Debian rootfs into $ROOTFS_DIR..."
sudo rm -rf "$ROOTFS_DIR"
mkdir -p "$ROOTFS_DIR"

sudo debootstrap \
    --arch="$DEBOOTSTRAP_ARCH" \
    --variant=minbase \
    --include=openssh-server,bash,coreutils,tmux,fuse3,libfuse3-3,systemd,systemd-sysv,dbus,iproute2 \
    "$DEBIAN_SUITE" "$ROOTFS_DIR" "$DEBIAN_MIRROR"

# Configure the rootfs inside chroot
echo "Configuring rootfs..."
sudo chroot "$ROOTFS_DIR" /bin/bash -c '
    # Enable sshd
    systemctl enable ssh

    # Create test user alice (uid=1000, gid=1000)
    groupadd -g 1000 alice
    useradd -m -u 1000 -g alice -s /bin/bash alice
    echo "alice:testpass" | chpasswd

    # Enable root login for dev/test SSH access
    sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
    echo "root:rootpass" | chpasswd

    # Generate SSH host keys
    ssh-keygen -A

    # Create motlie-vfs config directory
    mkdir -p /etc/motlie-vfs

    # Create systemd service for motlie-vfs-guest
    cat > /etc/systemd/system/motlie-vfs-guest.service << "SVCEOF"
[Unit]
Description=motlie-vfs guest filesystem mounter
After=local-fs.target
Before=ssh.service

[Service]
Type=simple
ExecStart=/usr/local/bin/motlie-vfs-guest /etc/motlie-vfs/mounts.yaml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
SVCEOF
    systemctl enable motlie-vfs-guest

    # Set hostname
    echo "motlie-guest" > /etc/hostname

    # Minimal fstab
    cat > /etc/fstab << "FSTABEOF"
# <device>  <mount>  <type>  <options>       <dump> <pass>
proc        /proc    proc    defaults        0      0
sysfs       /sys     sysfs   defaults        0      0
devtmpfs    /dev     devtmpfs defaults       0      0
FSTABEOF

    # Enable user_allow_other for FUSE — required so alice can access
    # the FUSE mount created by motlie-vfs-guest (runs as root)
    echo "user_allow_other" >> /etc/fuse.conf

    # Clean up apt cache
    apt-get clean
    rm -rf /var/lib/apt/lists/*
'

# Install the guest binary
echo "Installing motlie-vfs-guest binary..."
sudo cp "$GUEST_BINARY" "$ROOTFS_DIR/usr/local/bin/motlie-vfs-guest"
sudo chmod 755 "$ROOTFS_DIR/usr/local/bin/motlie-vfs-guest"

# Copy required shared libraries into the rootfs if not already present
echo "Ensuring shared library dependencies are satisfied..."
for lib in $(ldd "$GUEST_BINARY" | grep -o '/lib[^ ]*'); do
    target="$ROOTFS_DIR$lib"
    if [ ! -f "$target" ]; then
        echo "  copying $lib"
        sudo cp "$lib" "$target"
    fi
done

# Install overlay-init
echo "Installing overlay-init..."
sudo cp "$SCRIPT_DIR/overlay-init" "$ROOTFS_DIR/sbin/overlay-init"
sudo chmod 755 "$ROOTFS_DIR/sbin/overlay-init"

# Build squashfs
echo "Building squashfs image..."
sudo mksquashfs "$ROOTFS_DIR" "$ARTIFACTS/rootfs.squashfs" \
    -noappend -comp gzip -quiet

echo "Squashfs image: $ARTIFACTS/rootfs.squashfs ($(du -h "$ARTIFACTS/rootfs.squashfs" | cut -f1))"

# Clean up rootfs build dir
sudo rm -rf "$ROOTFS_DIR"

# ---------------------------------------------------------------------------
# Step 4: Build the ext4 overlay image with pre-seeded config
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Building ext4 overlay image ---"

# Create directory tree to seed into the ext4 image.
# This uses mkfs.ext4 -d to avoid needing sudo mount -o loop.
rm -rf "$OVERLAY_SEED"
mkdir -p "$OVERLAY_SEED/upper/etc/motlie-vfs"
mkdir -p "$OVERLAY_SEED/work"

# Copy the mount config
cp "$SCRIPT_DIR/mounts.yaml" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"

# Create alice's home directory structure and mount points.
# These directories must exist for the guest mounter to mount onto them.
# uid=1000 gid=1000 matches the alice user in the squashfs image.
mkdir -p "$OVERLAY_SEED/upper/home/alice/.ssh"
mkdir -p "$OVERLAY_SEED/upper/home/alice/workspace"
chmod 700 "$OVERLAY_SEED/upper/home/alice/.ssh"

# Build ext4 image from the seed directory (no sudo/mount needed)
truncate -s "$OVERLAY_SIZE" "$ARTIFACTS/overlay.ext4"
mkfs.ext4 -F -d "$OVERLAY_SEED" "$ARTIFACTS/overlay.ext4" -q

echo "Overlay image: $ARTIFACTS/overlay.ext4 ($OVERLAY_SIZE)"

# Clean up
rm -rf "$OVERLAY_SEED"

# ---------------------------------------------------------------------------
# Step 5: Fix artifact ownership
# ---------------------------------------------------------------------------
# build-guest.sh runs under sudo, so artifacts are root-owned.
# CH and launch-ch.sh run as the normal user and need read access to the
# disk images.  Chown everything back to the invoking user.
if [ -n "${SUDO_USER:-}" ]; then
    echo ""
    echo "--- Step 5: Fixing artifact ownership (SUDO_USER=$SUDO_USER) ---"
    chown -R "$SUDO_USER:$SUDO_USER" "$ARTIFACTS"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Build complete ==="
echo "Artifacts:"
ls -lh "$ARTIFACTS/"
echo ""
echo "Next: ./launch-ch.sh"
