#!/usr/bin/env bash
# build-guest.sh — Build a guest image set for the v1.1 multi-guest demo.
#
# Produces guest-specific artifacts under artifacts/<guest>/:
#   rootfs.squashfs        — Debian root with sshd, bash, fuse3, motlie-vfs-guest
#   overlay.ext4           — writable ext4 overlay seeded with mounts config
#   Image|vmlinux.bin      — CH-compatible kernel
#
# Usage:
#   ./build-guest.sh --guest alice
#   ./build-guest.sh --guest bob
#   ./build-guest.sh --guest alice --guest-binary /path/to/motlie-vfs-guest
#   ./build-guest.sh --guest bob --kernel skip
#
# Notes:
#   - No sudo required. Uses mmdebstrap --mode=unshare for rootless operation.
#   - The rootfs includes both alice (uid=1000) and bob (uid=1001) so either
#     guest overlay can boot the same binary/service set cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

GUEST_NAME="alice"
GUEST_BINARY=""
KERNEL_MODE="download"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest) GUEST_NAME="$2"; shift 2 ;;
        --guest-binary) GUEST_BINARY="$2"; shift 2 ;;
        --kernel) KERNEL_MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "$GUEST_NAME" in
    alice|bob) ;;
    *) echo "ERROR: --guest must be one of: alice, bob"; exit 1 ;;
esac

if [[ "$KERNEL_MODE" != "download" && "$KERNEL_MODE" != "build" && "$KERNEL_MODE" != "skip" ]]; then
    echo "ERROR: --kernel must be one of: download, build, skip"
    exit 1
fi

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

CH_KERNEL_RELEASE="ch-release-v6.16.9-20251112"
CH_KERNEL_URL="https://github.com/cloud-hypervisor/linux/releases/download/${CH_KERNEL_RELEASE}/${KERNEL_RELEASE_ASSET}"

for cmd in mmdebstrap mkfs.ext4 tar2sqfs; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found."
        echo "Install: sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring"
        exit 1
    fi
done

if [ ! -f /usr/share/keyrings/debian-archive-keyring.gpg ]; then
    echo "ERROR: Debian archive keyring not found."
    echo "Install: sudo apt install debian-archive-keyring"
    exit 1
fi

APPARMOR_USERNS=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)
if [ "$APPARMOR_USERNS" = "1" ]; then
    echo "ERROR: AppArmor restricts unprivileged user namespaces."
    echo "Fix:   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
    exit 1
fi

DEBIAN_SUITE="bookworm"
DEBIAN_MIRROR="http://deb.debian.org/debian"
ARTIFACTS="$SCRIPT_DIR/artifacts/$GUEST_NAME"
OVERLAY_SEED="/tmp/motlie-vfs-v11-overlay-${GUEST_NAME}-$$"
OVERLAY_SIZE="64M"
MOUNT_CONFIG="$SCRIPT_DIR/mounts.${GUEST_NAME}.yaml"

case "$GUEST_NAME" in
    alice)
        LOGIN_UID=1000
        LOGIN_GID=1000
        LOGIN_HOME="/home/alice"
        HOSTNAME="motlie-alice"
        ;;
    bob)
        LOGIN_UID=1001
        LOGIN_GID=1001
        LOGIN_HOME="/home/bob"
        HOSTNAME="motlie-bob"
        ;;
esac

if [ ! -f "$MOUNT_CONFIG" ]; then
    echo "ERROR: mount config not found at $MOUNT_CONFIG"
    exit 1
fi

echo "=== motlie-vfs v1.1 guest image builder ==="
echo "Guest:            $GUEST_NAME"
echo "Host arch:        $HOST_ARCH"
echo "Rust target:      $RUST_TARGET"
echo "Debian arch:      $DEBOOTSTRAP_ARCH"
echo "Kernel mode:      $KERNEL_MODE"
echo "Artifacts dir:    $ARTIFACTS"
echo "Mount config:     $MOUNT_CONFIG"
echo ""

mkdir -p "$ARTIFACTS"

if [ -z "$GUEST_BINARY" ]; then
    echo "--- Step 1: Building motlie-vfs-guest ($RUST_TARGET) ---"
    (cd "$WORKSPACE_ROOT" && cargo build --release \
        --features vsock,client \
        -p motlie-vfs --bin motlie-vfs-guest)
    GUEST_BINARY="$WORKSPACE_ROOT/target/release/motlie-vfs-guest"
    echo "Guest binary: $GUEST_BINARY"
    file "$GUEST_BINARY"
    echo ""
fi

if [ ! -f "$GUEST_BINARY" ]; then
    echo "ERROR: guest binary not found at $GUEST_BINARY"
    exit 1
fi

echo "--- Step 2: Kernel ($KERNEL_MODE) ---"
case "$KERNEL_MODE" in
    download)
        if [ -f "$ARTIFACTS/$KERNEL_IMAGE" ]; then
            echo "Kernel already exists: $ARTIFACTS/$KERNEL_IMAGE ($(du -h "$ARTIFACTS/$KERNEL_IMAGE" | cut -f1))"
        else
            echo "Downloading pre-built kernel from cloud-hypervisor/linux..."
            wget -q --show-progress -O "$ARTIFACTS/$KERNEL_IMAGE" "$CH_KERNEL_URL"
            echo "Kernel: $ARTIFACTS/$KERNEL_IMAGE ($(du -h "$ARTIFACTS/$KERNEL_IMAGE" | cut -f1))"
        fi
        ;;
    build)
        KERNEL_SRC="/tmp/motlie-vfs-kernel-${GUEST_NAME}-$$"
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
echo ""

echo "--- Step 3: Building Debian squashfs root image (rootless) ---"
GUEST_BINARY_ABS="$(realpath "$GUEST_BINARY")"
OVERLAY_INIT_ABS="$(realpath "$SCRIPT_DIR/overlay-init")"

mmdebstrap \
    --mode=unshare \
    --arch="$DEBOOTSTRAP_ARCH" \
    --variant=minbase \
    --format=squashfs \
    --keyring=/usr/share/keyrings/debian-archive-keyring.gpg \
    --include=openssh-server,bash,coreutils,tmux,fuse3,libfuse3-3,systemd,systemd-sysv,dbus,iproute2 \
    --customize-hook='chroot "$1" systemctl enable ssh' \
    --customize-hook='chroot "$1" groupadd -g 1000 alice' \
    --customize-hook='chroot "$1" useradd -m -u 1000 -g alice -s /bin/bash alice' \
    --customize-hook='echo "alice:testpass" | chroot "$1" chpasswd' \
    --customize-hook='chroot "$1" groupadd -g 1001 bob' \
    --customize-hook='chroot "$1" useradd -m -u 1001 -g bob -s /bin/bash bob' \
    --customize-hook='echo "bob:testpass" | chroot "$1" chpasswd' \
    --customize-hook='sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" "$1/etc/ssh/sshd_config"' \
    --customize-hook='echo "root:rootpass" | chroot "$1" chpasswd' \
    --customize-hook='chroot "$1" ssh-keygen -A' \
    --customize-hook='mkdir -p "$1/etc/motlie-vfs"' \
    --customize-hook='cat > "$1/etc/profile.d/dotenv.sh" << "DOTENVEOF"
if [ -f "$HOME/.env" ]; then
    set -a
    . "$HOME/.env"
    set +a
fi
DOTENVEOF' \
    --customize-hook='cat > "$1/etc/profile.d/tmux-auto.sh" << "TMUXEOF"
if [ -n "$SSH_CONNECTION" ] && [ -z "$TMUX" ] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$USER" 2>/dev/null; then
        echo "Attaching to existing tmux session..."
        sleep 1
        exec tmux attach-session -t "$USER"
    else
        printf "Start tmux session? [Y/n] (auto-yes in 3s) "
        if read -r -n 1 -t 3 answer; then
            echo
        else
            answer=Y
            echo
        fi
        case "$answer" in
            n|N) ;;
            *) exec tmux new-session -s "$USER" ;;
        esac
    fi
fi
TMUXEOF' \
    --customize-hook='cat > "$1/etc/systemd/system/motlie-vfs-guest.service" << "SVCEOF"
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
SVCEOF' \
    --customize-hook='chroot "$1" systemctl enable motlie-vfs-guest' \
    --customize-hook="echo \"$HOSTNAME\" > \"\$1/etc/hostname\"" \
    --customize-hook='cat > "$1/etc/motd" << "MOTDEOF"
                    _   _ _
  _ __ ___   ___ | |_| (_) ___
 | '"'"'_ ` _ \ / _ \| __| | |/ _ \
 | | | | | | (_) | |_| | |  __/
 |_| |_| |_|\___/ \__|_|_|\___|

v1.1 multi-guest demo
MOTDEOF' \
    --customize-hook='cat > "$1/etc/fstab" << "FSTABEOF"
proc        /proc    proc    defaults        0      0
sysfs       /sys     sysfs   defaults        0      0
devtmpfs    /dev     devtmpfs defaults       0      0
FSTABEOF' \
    --customize-hook='echo "user_allow_other" >> "$1/etc/fuse.conf"' \
    --customize-hook="upload $GUEST_BINARY_ABS /usr/local/bin/motlie-vfs-guest" \
    --customize-hook="chmod 755 \"\$1/usr/local/bin/motlie-vfs-guest\"" \
    --customize-hook="upload $OVERLAY_INIT_ABS /sbin/overlay-init" \
    --customize-hook="chmod 755 \"\$1/sbin/overlay-init\"" \
    --customize-hook='chroot "$1" apt-get clean' \
    --customize-hook='rm -rf "$1/var/lib/apt/lists"/*' \
    "$DEBIAN_SUITE" "$ARTIFACTS/rootfs.squashfs" "$DEBIAN_MIRROR"

echo "Squashfs image: $ARTIFACTS/rootfs.squashfs ($(du -h "$ARTIFACTS/rootfs.squashfs" | cut -f1))"
echo ""

echo "--- Step 4: Building ext4 overlay image ---"
rm -rf "$OVERLAY_SEED"
mkdir -p "$OVERLAY_SEED/upper/etc/motlie-vfs"
mkdir -p "$OVERLAY_SEED/work"
cp "$MOUNT_CONFIG" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"

mkdir -p "$OVERLAY_SEED/upper${LOGIN_HOME}/.ssh"
mkdir -p "$OVERLAY_SEED/upper/workspace"
chmod 700 "$OVERLAY_SEED/upper${LOGIN_HOME}/.ssh"

truncate -s "$OVERLAY_SIZE" "$ARTIFACTS/overlay.ext4"
mkfs.ext4 -F -d "$OVERLAY_SEED" "$ARTIFACTS/overlay.ext4" -q
rm -rf "$OVERLAY_SEED"

echo "Overlay image: $ARTIFACTS/overlay.ext4 ($OVERLAY_SIZE)"
echo ""
echo "=== Build complete ==="
echo "Guest:     $GUEST_NAME"
echo "Login:     uid=$LOGIN_UID gid=$LOGIN_GID home=$LOGIN_HOME"
echo "Artifacts: $ARTIFACTS"
ls -lh "$ARTIFACTS/"
echo ""
echo "Next: ./launch-ch.sh --guest $GUEST_NAME"
