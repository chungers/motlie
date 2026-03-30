#!/usr/bin/env bash
# build-guest.sh — Build the squashfs + ext4 guest image for Cloud Hypervisor.
#
# Produces:
#   artifacts/rootfs.squashfs  — read-only Alpine root with sshd, bash, motlie-vfs-guest
#   artifacts/overlay.ext4     — writable ext4 overlay pre-seeded with mounts.yaml
#   artifacts/vmlinux.bin      — CH-compatible kernel (built separately, see README)
#
# Prerequisites (Linux host or Docker):
#   - squashfs-tools (mksquashfs)
#   - e2fsprogs (mkfs.ext4, mke2fs)
#   - wget or curl
#   - sudo (for chroot-based rootfs assembly)
#
# For cross-compiling the guest binary from macOS:
#   - cargo install cross (uses Docker)
#   - or: brew install filosottile/musl-cross/musl-cross
#
# Usage:
#   ./build-guest.sh [--guest-binary /path/to/motlie-vfs-guest]
#
# If --guest-binary is not provided, the script will attempt to cross-compile
# it from the workspace root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS="$SCRIPT_DIR/artifacts"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
GUEST_BINARY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest-binary) GUEST_BINARY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALPINE_VERSION="3.21"
ALPINE_MIRROR="https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION}"
ROOTFS_DIR="/tmp/motlie-vfs-rootfs-$$"
OVERLAY_SEED="/tmp/motlie-vfs-overlay-seed-$$"
OVERLAY_SIZE="64M"

echo "=== motlie-vfs guest image builder ==="
echo "Artifacts dir: $ARTIFACTS"
echo "Workspace root: $WORKSPACE_ROOT"

mkdir -p "$ARTIFACTS"

# ---------------------------------------------------------------------------
# Step 1: Cross-compile the guest binary (if not provided)
# ---------------------------------------------------------------------------
if [ -z "$GUEST_BINARY" ]; then
    echo ""
    echo "--- Step 1: Cross-compiling motlie-vfs-guest for x86_64-linux-musl ---"

    # Prefer `cross` if available (Docker-based, works from macOS)
    if command -v cross &>/dev/null; then
        echo "Using 'cross' for musl cross-compilation"
        (cd "$WORKSPACE_ROOT" && cross build --release \
            --target x86_64-unknown-linux-musl \
            -p motlie-vfs --bin motlie-vfs-guest)
        GUEST_BINARY="$WORKSPACE_ROOT/target/x86_64-unknown-linux-musl/release/motlie-vfs-guest"
    else
        echo "Using native cargo (requires musl toolchain installed)"
        echo "Install cross for easier cross-compilation: cargo install cross"
        rustup target add x86_64-unknown-linux-musl 2>/dev/null || true
        (cd "$WORKSPACE_ROOT" && cargo build --release \
            --target x86_64-unknown-linux-musl \
            -p motlie-vfs --bin motlie-vfs-guest)
        GUEST_BINARY="$WORKSPACE_ROOT/target/x86_64-unknown-linux-musl/release/motlie-vfs-guest"
    fi

    echo "Guest binary: $GUEST_BINARY"
    file "$GUEST_BINARY"
fi

if [ ! -f "$GUEST_BINARY" ]; then
    echo "ERROR: guest binary not found at $GUEST_BINARY"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Build the Alpine squashfs root image
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Building Alpine squashfs root image ---"

# Download apk.static if not cached
APK_STATIC="$ARTIFACTS/apk.static"
if [ ! -f "$APK_STATIC" ]; then
    echo "Downloading apk.static..."
    wget -q -O "$APK_STATIC" \
        "https://gitlab.alpinelinux.org/api/v4/projects/5/packages/generic/v2.14.6/x86_64/apk.static"
    chmod +x "$APK_STATIC"
fi

# Bootstrap Alpine rootfs
echo "Bootstrapping Alpine rootfs into $ROOTFS_DIR..."
sudo rm -rf "$ROOTFS_DIR"
mkdir -p "$ROOTFS_DIR"

sudo "$APK_STATIC" \
    --arch x86_64 \
    -X "$ALPINE_MIRROR/main" \
    -X "$ALPINE_MIRROR/community" \
    -U --allow-untrusted \
    --root "$ROOTFS_DIR" \
    --initdb \
    add alpine-base openssh-server bash util-linux coreutils

# Configure the rootfs inside chroot
echo "Configuring rootfs..."
sudo chroot "$ROOTFS_DIR" /bin/sh -c '
    # Enable services for OpenRC
    rc-update add sshd default
    rc-update add devfs sysinit
    rc-update add procfs sysinit
    rc-update add sysfs sysinit

    # Create test user alice (uid=1000, gid=1000)
    addgroup -g 1000 alice
    adduser -D -u 1000 -G alice -h /home/alice -s /bin/bash alice
    echo "alice:testpass" | chpasswd

    # Enable root login for dev/test SSH access
    sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
    echo "root:rootpass" | chpasswd

    # Generate SSH host keys
    ssh-keygen -A

    # Create motlie-vfs config directory
    mkdir -p /etc/motlie-vfs
'

# Install the guest binary
echo "Installing motlie-vfs-guest binary..."
sudo cp "$GUEST_BINARY" "$ROOTFS_DIR/usr/local/bin/motlie-vfs-guest"
sudo chmod 755 "$ROOTFS_DIR/usr/local/bin/motlie-vfs-guest"

# Install overlay-init
echo "Installing overlay-init..."
sudo cp "$SCRIPT_DIR/overlay-init" "$ROOTFS_DIR/sbin/overlay-init"
sudo chmod 755 "$ROOTFS_DIR/sbin/overlay-init"

# Clean up package cache to reduce image size
sudo rm -rf "$ROOTFS_DIR/var/cache/apk"/*

# Build squashfs
echo "Building squashfs image..."
sudo mksquashfs "$ROOTFS_DIR" "$ARTIFACTS/rootfs.squashfs" \
    -noappend -comp zstd -quiet

echo "Squashfs image: $ARTIFACTS/rootfs.squashfs ($(du -h "$ARTIFACTS/rootfs.squashfs" | cut -f1))"

# Clean up rootfs build dir
sudo rm -rf "$ROOTFS_DIR"

# ---------------------------------------------------------------------------
# Step 3: Build the ext4 overlay image with pre-seeded config
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Building ext4 overlay image ---"

# Create directory tree to seed into the ext4 image.
# This uses mkfs.ext4 -d to avoid needing sudo mount -o loop.
rm -rf "$OVERLAY_SEED"
mkdir -p "$OVERLAY_SEED/upper/etc/motlie-vfs"
mkdir -p "$OVERLAY_SEED/work"

# Copy the mount config
cp "$SCRIPT_DIR/mounts.yaml" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"

# Create alice's home directory structure
mkdir -p "$OVERLAY_SEED/upper/home/alice"

# Build ext4 image from the seed directory (no sudo/mount needed)
truncate -s "$OVERLAY_SIZE" "$ARTIFACTS/overlay.ext4"
mkfs.ext4 -F -d "$OVERLAY_SEED" "$ARTIFACTS/overlay.ext4" -q

echo "Overlay image: $ARTIFACTS/overlay.ext4 ($OVERLAY_SIZE)"

# Clean up
rm -rf "$OVERLAY_SEED"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Build complete ==="
echo "Artifacts:"
ls -lh "$ARTIFACTS/"
echo ""
echo "Next: build or download a CH-compatible kernel (see README.md)"
echo "Then: ./launch-ch.sh"
