#!/usr/bin/env bash
# launch-ch.sh — Launch one guest VM for the v1.1 multi-guest demo.
#
# Usage:
#   ./launch-ch.sh --guest alice
#   ./launch-ch.sh --guest bob
#   ./launch-ch.sh --guest alice --no-net
#   ./launch-ch.sh --guest alice --overlay-size 2G
#   ./launch-ch.sh --guest alice --cloud-init-dir /tmp/alice-cloud-init
#
# Shared built artifacts:
#   artifacts/base/rootfs.squashfs
#   artifacts/base/Image|vmlinux.bin
#
# Per-guest writable runtime overlays are created on each launch under:
#   ${RUNTIME_ROOT:-/tmp/motlie-vfs-v11-runtime}/<guest>/overlay.ext4

set -euo pipefail

copy_overlay_tree() {
    local source_dir="$1"
    local dest_dir="$2"

    if [ -d "$source_dir" ]; then
        mkdir -p "$dest_dir"
        cp -a "$source_dir"/. "$dest_dir"/
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64) KERNEL_IMAGE="vmlinux.bin" ;;
    aarch64) KERNEL_IMAGE="Image" ;;
    *) echo "ERROR: unsupported host architecture: $HOST_ARCH"; exit 1 ;;
esac

GUEST_NAME="alice"
USE_NET=true
OVERLAY_SIZE="${OVERLAY_SIZE:-2G}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/motlie-vfs-v11-runtime}"
CLOUD_INIT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest) GUEST_NAME="$2"; shift 2 ;;
        --overlay-size) OVERLAY_SIZE="$2"; shift 2 ;;
        --cloud-init-dir) CLOUD_INIT_DIR="$2"; shift 2 ;;
        --no-net) USE_NET=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "$GUEST_NAME" in
    alice)
        BASE_ARTIFACTS="$SCRIPT_DIR/artifacts/base"
        API_SOCKET="/tmp/motlie-vfs-alice-api.sock"
        VSOCK_SOCKET="/tmp/motlie-vfs-alice.vsock"
        CID=3
        HOST_IP="192.168.249.1"
        GUEST_IP="192.168.249.2"
        MAC="12:34:56:78:90:aa"
        SSH_USER="alice"
        GUEST_HOSTNAME="motlie-alice"
        LOGIN_HOME="/home/alice"
        MOUNT_CONFIG="$SCRIPT_DIR/mounts.alice.yaml"
        GUEST_OVERLAY_CONTENT="$SCRIPT_DIR/overlay.d/alice"
        ;;
    bob)
        BASE_ARTIFACTS="$SCRIPT_DIR/artifacts/base"
        API_SOCKET="/tmp/motlie-vfs-bob-api.sock"
        VSOCK_SOCKET="/tmp/motlie-vfs-bob.vsock"
        CID=4
        HOST_IP="192.168.250.1"
        GUEST_IP="192.168.250.2"
        MAC="12:34:56:78:90:bb"
        SSH_USER="bob"
        GUEST_HOSTNAME="motlie-bob"
        LOGIN_HOME="/home/bob"
        MOUNT_CONFIG="$SCRIPT_DIR/mounts.bob.yaml"
        GUEST_OVERLAY_CONTENT="$SCRIPT_DIR/overlay.d/bob"
        ;;
    *)
        echo "ERROR: --guest must be one of: alice, bob"
        exit 1
        ;;
esac

if [ ! -f "$BASE_ARTIFACTS/$KERNEL_IMAGE" ]; then
    echo "ERROR: $BASE_ARTIFACTS/$KERNEL_IMAGE not found."
    echo "Run ./build-guest.sh first."
    exit 1
fi
if [ ! -f "$BASE_ARTIFACTS/rootfs.squashfs" ]; then
    echo "ERROR: $BASE_ARTIFACTS/rootfs.squashfs not found."
    echo "Run ./build-guest.sh first."
    exit 1
fi
if [ ! -f "$MOUNT_CONFIG" ]; then
    echo "ERROR: mount config not found at $MOUNT_CONFIG"
    exit 1
fi
if ! command -v mkfs.ext4 >/dev/null 2>&1; then
    echo "ERROR: mkfs.ext4 not found. Install: sudo apt install e2fsprogs"
    exit 1
fi
if [ -n "$CLOUD_INIT_DIR" ]; then
    if [ ! -d "$CLOUD_INIT_DIR" ]; then
        echo "ERROR: cloud-init dir not found at $CLOUD_INIT_DIR"
        exit 1
    fi
    if [ ! -f "$CLOUD_INIT_DIR/user-data" ]; then
        echo "ERROR: cloud-init dir is missing user-data at $CLOUD_INIT_DIR/user-data"
        exit 1
    fi
    if [ ! -f "$CLOUD_INIT_DIR/meta-data" ]; then
        echo "ERROR: cloud-init dir is missing meta-data at $CLOUD_INIT_DIR/meta-data"
        exit 1
    fi
fi

COMMON_OVERLAY_CONTENT="$SCRIPT_DIR/overlay.d/common"
RUNTIME_DIR="$RUNTIME_ROOT/$GUEST_NAME"
RUNTIME_OVERLAY="$RUNTIME_DIR/overlay.ext4"
OVERLAY_SEED="$RUNTIME_DIR/seed"

rm -rf "$OVERLAY_SEED"
mkdir -p "$OVERLAY_SEED/upper/etc/motlie-vfs"
mkdir -p "$OVERLAY_SEED/work"

if [ -n "$CLOUD_INIT_DIR" ]; then
    mkdir -p "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud"
    cp "$CLOUD_INIT_DIR/user-data" "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud/user-data"
    cp "$CLOUD_INIT_DIR/meta-data" "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud/meta-data"
    if [ -f "$CLOUD_INIT_DIR/network-config" ]; then
        cp "$CLOUD_INIT_DIR/network-config" "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud/network-config"
    fi
else
    cp "$MOUNT_CONFIG" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"
    printf '%s\n' "$GUEST_HOSTNAME" > "$OVERLAY_SEED/upper/etc/hostname"
    mkdir -p "$OVERLAY_SEED/upper${LOGIN_HOME}/.ssh"
    mkdir -p "$OVERLAY_SEED/upper/workspace"
    chmod 700 "$OVERLAY_SEED/upper${LOGIN_HOME}/.ssh"
fi

copy_overlay_tree "$COMMON_OVERLAY_CONTENT" "$OVERLAY_SEED/upper"
copy_overlay_tree "$GUEST_OVERLAY_CONTENT" "$OVERLAY_SEED/upper"
mkdir -p "$RUNTIME_DIR"
rm -f "$RUNTIME_OVERLAY"
truncate -s "$OVERLAY_SIZE" "$RUNTIME_OVERLAY"
mkfs.ext4 -F -d "$OVERLAY_SEED" "$RUNTIME_OVERLAY" -q
rm -rf "$OVERLAY_SEED"

if [ ! -e /dev/vhost-vsock ]; then
    echo "Loading vhost_vsock kernel module..."
    sudo modprobe vhost_vsock || {
        echo "ERROR: could not load vhost_vsock module."
        exit 1
    }
fi

CMDLINE="console=ttyS0 root=/dev/vda rootfstype=squashfs ro init=/sbin/overlay-init overlay_root=vdb"
if [ -n "$CLOUD_INIT_DIR" ]; then
    CMDLINE="$CMDLINE ds=nocloud;s=file:///var/lib/cloud/seed/nocloud/"
fi
if $USE_NET; then
    CMDLINE="$CMDLINE ip=${GUEST_IP}::${HOST_IP}:255.255.255.0::eth0:off"
fi

CH_ARGS=(
    --api-socket "$API_SOCKET"
    --kernel "$BASE_ARTIFACTS/$KERNEL_IMAGE"
    --cmdline "$CMDLINE"
    --cpus boot=2
    --memory size=512M
    --vsock "cid=$CID,socket=$VSOCK_SOCKET"
    --serial tty
    --console off
)

DISK_ARGS=(
    "path=$BASE_ARTIFACTS/rootfs.squashfs,readonly=on"
    "path=$RUNTIME_OVERLAY"
)

if $USE_NET; then
    CH_ARGS+=(--net "tap=,mac=$MAC,ip=$HOST_IP,mask=255.255.255.0")
fi

rm -f "$API_SOCKET" "$VSOCK_SOCKET"

echo "=== Launching Cloud Hypervisor ($GUEST_NAME) ==="
echo "  Kernel:    $BASE_ARTIFACTS/$KERNEL_IMAGE"
echo "  Squashfs:  $BASE_ARTIFACTS/rootfs.squashfs (vda, ro)"
echo "  Overlay:   $RUNTIME_OVERLAY (vdb, rw runtime)"
echo "  Size:      $OVERLAY_SIZE"
if [ -n "$CLOUD_INIT_DIR" ]; then
    echo "  CloudInit: $CLOUD_INIT_DIR (seeded into /var/lib/cloud/seed/nocloud)"
fi
echo "  vsock:     CID $CID, socket $VSOCK_SOCKET"
if $USE_NET; then
    echo "  Network:   TAP, guest $GUEST_IP, host $HOST_IP"
    echo "  SSH:       ssh $SSH_USER@$GUEST_IP (password: testpass)"
else
    echo "  Network:   disabled (--no-net)"
fi
echo "  Lifetime:  recreated on every launch"
echo "  API:       $API_SOCKET"
echo ""
echo "To stop: curl --unix-socket $API_SOCKET -X PUT http://localhost/api/v1/vm.shutdown"
echo ""

cloud-hypervisor "${CH_ARGS[@]}" --disk "${DISK_ARGS[@]}"
