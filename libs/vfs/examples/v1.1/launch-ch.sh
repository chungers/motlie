#!/usr/bin/env bash
# launch-ch.sh — Launch one guest VM for the v1.1 multi-guest demo.
#
# Usage:
#   ./launch-ch.sh --guest alice
#   ./launch-ch.sh --guest bob
#   ./launch-ch.sh --guest alice --no-net
#
# Guest-specific defaults:
#   alice -> CID 3, socket /tmp/motlie-vfs-alice.vsock, api /tmp/motlie-vfs-alice-api.sock
#   bob   -> CID 4, socket /tmp/motlie-vfs-bob.vsock,   api /tmp/motlie-vfs-bob-api.sock

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64) KERNEL_IMAGE="vmlinux.bin" ;;
    aarch64) KERNEL_IMAGE="Image" ;;
    *) echo "ERROR: unsupported host architecture: $HOST_ARCH"; exit 1 ;;
esac

GUEST_NAME="alice"
USE_NET=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest) GUEST_NAME="$2"; shift 2 ;;
        --no-net) USE_NET=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "$GUEST_NAME" in
    alice)
        ARTIFACTS="$SCRIPT_DIR/artifacts/alice"
        API_SOCKET="/tmp/motlie-vfs-alice-api.sock"
        VSOCK_SOCKET="/tmp/motlie-vfs-alice.vsock"
        CID=3
        HOST_IP="192.168.249.1"
        GUEST_IP="192.168.249.2"
        MAC="12:34:56:78:90:aa"
        SSH_USER="alice"
        ;;
    bob)
        ARTIFACTS="$SCRIPT_DIR/artifacts/bob"
        API_SOCKET="/tmp/motlie-vfs-bob-api.sock"
        VSOCK_SOCKET="/tmp/motlie-vfs-bob.vsock"
        CID=4
        HOST_IP="192.168.250.1"
        GUEST_IP="192.168.250.2"
        MAC="12:34:56:78:90:bb"
        SSH_USER="bob"
        ;;
    *)
        echo "ERROR: --guest must be one of: alice, bob"
        exit 1
        ;;
esac

for f in "$KERNEL_IMAGE" rootfs.squashfs overlay.ext4; do
    if [ ! -f "$ARTIFACTS/$f" ]; then
        echo "ERROR: $ARTIFACTS/$f not found."
        echo "Run ./build-guest.sh --guest $GUEST_NAME first."
        exit 1
    fi
done

if [ ! -e /dev/vhost-vsock ]; then
    echo "Loading vhost_vsock kernel module..."
    sudo modprobe vhost_vsock || {
        echo "ERROR: could not load vhost_vsock module."
        exit 1
    }
fi

CMDLINE="console=hvc0 root=/dev/vda rootfstype=squashfs ro init=/sbin/overlay-init overlay_root=vdb"
if $USE_NET; then
    CMDLINE="$CMDLINE ip=${GUEST_IP}::${HOST_IP}:255.255.255.0::eth0:off"
fi

CH_ARGS=(
    --api-socket "$API_SOCKET"
    --kernel "$ARTIFACTS/$KERNEL_IMAGE"
    --cmdline "$CMDLINE"
    --cpus boot=2
    --memory size=512M
    --disk "path=$ARTIFACTS/rootfs.squashfs,readonly=on"
           "path=$ARTIFACTS/overlay.ext4"
    --vsock "cid=$CID,socket=$VSOCK_SOCKET"
    --serial tty
    --console off
)

if $USE_NET; then
    CH_ARGS+=(--net "tap=,mac=$MAC,ip=$HOST_IP,mask=255.255.255.0")
fi

rm -f "$API_SOCKET" "$VSOCK_SOCKET"

echo "=== Launching Cloud Hypervisor ($GUEST_NAME) ==="
echo "  Kernel:    $ARTIFACTS/$KERNEL_IMAGE"
echo "  Squashfs:  $ARTIFACTS/rootfs.squashfs (vda, ro)"
echo "  Overlay:   $ARTIFACTS/overlay.ext4 (vdb, rw)"
echo "  vsock:     CID $CID, socket $VSOCK_SOCKET"
if $USE_NET; then
    echo "  Network:   TAP, guest $GUEST_IP, host $HOST_IP"
    echo "  SSH:       ssh $SSH_USER@$GUEST_IP (password: testpass)"
else
    echo "  Network:   disabled (--no-net)"
fi
echo "  API:       $API_SOCKET"
echo ""
echo "To stop: curl --unix-socket $API_SOCKET -X PUT http://localhost/api/v1/vm.shutdown"
echo ""

cloud-hypervisor "${CH_ARGS[@]}"
